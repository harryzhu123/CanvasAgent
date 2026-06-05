# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class ImageEditTokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


@ray.remote(num_gpus=0)  # GPU pinned manually via CUDA_VISIBLE_DEVICES
class ImageEditExecutionWorker:
    """Worker for executing image editing with GPU model."""

    def __init__(self, model_name: str, enable_global_rate_limit: bool = True, rate_limit: int = 10, gpu_id: str = "0"):
        """Initialize image editing worker.

        Args:
            model_name: Name or path of the diffusion model to load
            enable_global_rate_limit: Whether to enable rate limiting
            rate_limit: Maximum number of concurrent requests
            gpu_id: Physical GPU ID to pin this worker to
        """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.model_name = model_name
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.model = None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return ImageEditTokenBucketWorker.options(name="img-edit-rate-limiter", get_if_exists=True).remote(rate_limit)

    def _load_model(self):
        """Lazy load the model to avoid loading during initialization."""
        if self.model is not None:
            return

        try:
            import torch
            from diffusers import Flux2KleinPipeline

            logger.info(f"Loading FLUX.2 Klein model: {self.model_name}")
            self.model = Flux2KleinPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            # Load directly to GPU for faster inference (requires more VRAM)
            self.model.to("cuda", torch.bfloat16)
            self.model.set_progress_bar_config(disable=True)
            logger.info(f"Model {self.model_name} loaded successfully on GPU")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def ping(self):
        """Health check method."""
        return True

    def execute(self, image, prompt: str, masked_image=None, **kwargs) -> Any:
        """Execute image editing with optional rate limiting.

        Args:
            image: Input PIL Image to edit
            prompt: Text prompt for editing instruction
            masked_image: Optional mask PIL Image for localized editing
            **kwargs: Additional parameters for the model

        Returns:
            Edited PIL Image
        """
        # Lazy load model on first use
        self._load_model()

        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return self._edit_image(image, prompt, masked_image=masked_image, **kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing image editing: {e}")
                    raise
        else:
            return self._edit_image(image, prompt, masked_image=masked_image, **kwargs)

    def _edit_image(self, image, prompt: str, masked_image=None, **kwargs):
        """Internal method to edit image using the FLUX.2 Klein model.

        When masked_image is provided, performs pseudo-inpainting:
        1. Preprocess: fill the masked region with black, keep the rest unchanged.
        2. Pass the preprocessed image to the pipeline with the edit prompt.
        3. Composite: blend the generated result back with the original using the mask.
        """
        import torch
        import numpy as np
        from PIL import Image

        # Ensure image is PIL Image or a list of PIL Images in RGB mode.
        # FLUX.2 Klein supports multiple reference images via image=[...].
        is_multi_image = isinstance(image, list)
        if is_multi_image:
            if not image:
                raise ValueError("Input image list must not be empty")
            if not all(isinstance(img, Image.Image) for img in image):
                raise ValueError("All input images must be PIL Images")
            images = [img.convert("RGB") for img in image]
        else:
            if not isinstance(image, Image.Image):
                raise ValueError("Input image must be a PIL Image or a list of PIL Images")
            image = image.convert('RGB')

        # Preprocess: apply mask to create the pipeline input
        if masked_image is not None:
            if is_multi_image:
                raise ValueError("Masked image editing is only supported for single-image input")
            if not isinstance(masked_image, Image.Image):
                raise ValueError("Masked image must be a PIL Image")
            # Convert mask to grayscale, resize to match image
            mask_l = masked_image.convert('L').resize(image.size, Image.BILINEAR)
            # Fill masked region (white=255 in mask) with black, keep the rest
            img_array = np.array(image)
            mask_array = np.array(mask_l) / 255.0  # normalize to 0-1
            # mask=1 means region to edit -> fill with black
            input_array = (img_array * (1.0 - mask_array[..., None])).astype(np.uint8)
            pipeline_input = Image.fromarray(input_array)
        elif is_multi_image:
            pipeline_input = images
        else:
            pipeline_input = image

        # Extract generation parameters (FLUX.2 Klein specific defaults)
        num_inference_steps = kwargs.get("num_inference_steps", 4)
        guidance_scale = kwargs.get("guidance_scale", 3.5)
        negative_prompt = kwargs.get("negative_prompt", "")
        seed = kwargs.get("seed", None)

        with torch.no_grad():
            generator = None
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)

            pipe_kwargs = {
                "prompt": prompt,
                "image": pipeline_input,
                "guidance_scale": 1.0,  # FLUX.2 Klein is step-wise distilled, CFG not supported
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }

            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt

            output = self.model(**pipe_kwargs)
            edited_image = output.images[0]

            # Post-process: composite edited result with original using the mask
            if masked_image is not None:
                edited_image = edited_image.resize(image.size, Image.LANCZOS)
                # mask white=edit region: take from edited; mask black=keep region: take from original
                edited_image = Image.composite(edited_image, image, mask_l)

        return edited_image


def init_image_edit_execution_pool(
    model_name: str, num_workers: int, enable_global_rate_limit: bool = True, rate_limit: int = 10, num_gpus: float = 0, gpu_id: str = "0"
):
    """Initialize image editing execution pool.

    Uses named Ray actors with get_if_exists=True so that multiple AgentLoopWorkers
    share the same global pool of GPU workers instead of each creating their own.

    Args:
        model_name: Name or path of the diffusion model
        num_workers: Number of parallel workers
        enable_global_rate_limit: Whether to enable rate limiting
        rate_limit: Maximum number of concurrent requests
        num_gpus: Ray GPU resource (0 = manually pinned via CUDA_VISIBLE_DEVICES)
        gpu_id: Physical GPU ID to pin workers to

    Returns:
        List of Ray actor handles for image editing workers
    """
    workers = []
    for i in range(num_workers):
        actor_name = f"img-edit-worker-{i}"
        worker = (
            ImageEditExecutionWorker.options(name=actor_name, get_if_exists=True, num_gpus=num_gpus)
            .remote(model_name=model_name, enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, gpu_id=gpu_id)
        )
        workers.append(worker)
        logger.info(f"Created/reused image editing worker {i+1}/{num_workers} (name={actor_name}, gpu={gpu_id})")

    return workers


class ImageEditTool(BaseTool):
    """A tool for editing images based on text instructions using FLUX.2 Klein model.

    This tool provides image editing functionality with rate limiting and concurrent
    execution support through Ray. It supports various editing tasks including:
    - Global image editing based on text prompts
    - Localized editing using mask images
    - Adding/removing objects
    - Changing styles and attributes

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image editing
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageEditTool
            config:
              type: native
              model_name: "/nfsdata4/zhuhairui/FLUX.2-klein-4B"
              num_workers: 2           # Number of parallel workers
              num_gpus_per_worker: 0.5 # GPU allocation per worker
              rate_limit: 10           # Max concurrent requests
              enable_global_rate_limit: true
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageEditTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "ImageEdit",
                    "description": "Modifies an existing image (adding/removing objects, changing styles). Requires a reference image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "The ID of the image to modify."
                            },
                            "edit_prompt": {
                                "type": "string",
                                "description": "Instructions for the specific changes desired."
                            },
                            "masked_image": {
                                "type": "string",
                                "description": "Optional: A mask image ID to localize the edit region."
                            },
                            "num_inference_steps": {
                                "type": "integer",
                                "description": "Number of denoising steps (default: 4)"
                            },
                            "guidance_scale": {
                                "type": "number",
                                "description": "How closely to follow the prompt (default: 3.5)"
                            }
                        },
                        "required": ["edit_prompt"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.model_name = config.get("model_name", "/nfsdata4/zhuhairui/FLUX.2-klein-4B")
        self.num_workers = config.get("num_workers", 2)
        self.num_gpus_per_worker = config.get("num_gpus_per_worker", 0)
        self.gpu_id = config.get("gpu_id", "0")
        self.rate_limit = config.get("rate_limit", 10)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Initialize execution pool
        self.execution_pool = init_image_edit_execution_pool(
            model_name=self.model_name,
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            num_gpus=self.num_gpus_per_worker,
            gpu_id=self.gpu_id,
        )

        # Worker index for round-robin load balancing
        self.worker_index = 0

        logger.info(
            f"Initialized ImageEditTool with model={self.model_name}, "
            f"num_workers={self.num_workers}, rate_limit={self.rate_limit}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional creation parameters. May contain 'image' key,
                or 'create_kwargs' containing it.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs (optional - can be provided via shared_tool_outputs)
        image = kwargs.get("image")

        self._instance_dict[instance_id] = {
            "image": image,  # Store initial image from create_kwargs
            "images": [],  # Store edited images
            "prompts": [],  # Store prompts history
            "original_images": [],  # Store original images
            "reward": 0.0,
        }

        logger.debug(f"Created image editing instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image editing tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - edit_prompt: Text instruction for how to edit the image (required)
                - image: Reference to image in shared_tool_outputs (e.g., "img_1")
                - masked_image: Optional reference to mask image for localized editing
                - num_inference_steps: Number of denoising steps (default: 4)
                - guidance_scale: How closely to follow the prompt (default: 3.5)
                - negative_prompt: What to avoid in the edited image (optional)
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the edited image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Extract parameters (FLUX.2 Klein specific defaults)
        # Support both 'edit_prompt' (schema name) and 'prompt' (legacy name)
        edit_prompt = parameters.get("edit_prompt") or parameters.get("prompt")
        image_param = parameters.get("image") or parameters.get("image_ref")  # Support both 'image' and 'image_ref'
        masked_image_param = parameters.get("masked_image")  # Optional mask parameter
        num_inference_steps = parameters.get("num_inference_steps", 4)  # FLUX.2 default
        guidance_scale = parameters.get("guidance_scale", 3.5)  # FLUX.2 default
        negative_prompt = parameters.get("negative_prompt", "")
        seed = parameters.get("seed", None)

        # Get shared data
        shared_tool_outputs = kwargs.get("shared_tool_outputs", {})
        image_data = kwargs.get("image_data") or []

        # Get image from various sources (priority order):
        # 1. shared_tool_outputs (if image parameter is provided)
        # 2. instance_dict (from create_kwargs)
        # 3. image_data[0] (first image in the list)
        image = None
        if image_param and image_param in shared_tool_outputs:
            image = shared_tool_outputs[image_param]
        elif self._instance_dict[instance_id]["image"] is not None:
            image = self._instance_dict[instance_id]["image"]
        elif image_data:
            image = image_data[0]

        # Get masked_image if provided (optional for localized editing)
        masked_image = None
        if masked_image_param and masked_image_param in shared_tool_outputs:
            masked_image = shared_tool_outputs[masked_image_param]

        # Validate edit_prompt
        if not edit_prompt or not isinstance(edit_prompt, str):
            error_msg = "Error: 'edit_prompt' is missing or not a string in parameters."
            logger.error(f"[ImageEditTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageEditTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Select worker using round-robin load balancing
        worker = self.execution_pool[self.worker_index % len(self.execution_pool)]
        self.worker_index += 1

        try:
            # Execute image editing
            logger.info(f"Editing image with prompt: {edit_prompt[:50]}...")
            edited_image = await worker.execute.remote(
                image=image,
                prompt=edit_prompt,
                masked_image=masked_image,  # Pass masked_image for localized editing
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                seed=seed,
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["images"].append(edited_image)
            self._instance_dict[instance_id]["prompts"].append(edit_prompt)
            self._instance_dict[instance_id]["original_images"].append(image)

            # Prepare response
            response_text = f"Edited image with instruction: {edit_prompt}"

            # Metrics
            metrics = {
                "success": True,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "worker_id": self.worker_index - 1,
                "seed": seed,
            }

            logger.info(f"Image editing successful for instance {instance_id}")
            return ToolResponse(image=[edited_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image editing failed: {e}"
            logger.error(f"[ImageEditTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully edited images).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_images = len(self._instance_dict[instance_id]["images"])
        return float(num_images)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_images = len(self._instance_dict[instance_id]["images"])
            logger.debug(f"Releasing instance {instance_id} with {num_images} edited images")
            del self._instance_dict[instance_id]
