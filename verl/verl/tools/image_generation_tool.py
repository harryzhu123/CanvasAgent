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
class TokenBucketWorker:
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
class ImageGenExecutionWorker:
    """Worker for executing image generation with GPU model."""

    def __init__(self, model_name: str, enable_global_rate_limit: bool = True, rate_limit: int = 10, gpu_id: str = "0"):
        """Initialize image generation worker.

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
        return TokenBucketWorker.options(name="img-gen-rate-limiter", get_if_exists=True).remote(rate_limit)

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

    def execute(self, prompt: str, **kwargs) -> Any:
        """Execute image generation with optional rate limiting.

        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters for the model (e.g., num_inference_steps, guidance_scale)

        Returns:
            Generated PIL Image
        """
        # Lazy load model on first use
        self._load_model()

        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return self._generate_image(prompt, **kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing image generation: {e}")
                    raise
        else:
            return self._generate_image(prompt, **kwargs)

    def _generate_image(self, prompt: str, **kwargs):
        """Internal method to generate image using the FLUX.2 Klein model."""
        import torch

        # Extract generation parameters (FLUX.2 Klein specific defaults)
        num_inference_steps = kwargs.get("num_inference_steps", 4)  # FLUX.2 Klein step-distilled default
        negative_prompt = kwargs.get("negative_prompt", "")
        height = kwargs.get("height", 1024)
        width = kwargs.get("width", 1024)
        seed = kwargs.get("seed", None)

        with torch.no_grad():
            # Set up generator with optional seed
            generator = None
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)

            pipe_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": 1.0,  # FLUX.2 Klein is step-wise distilled, CFG not supported
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }

            # Add negative_prompt if not empty
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt

            output = self.model(**pipe_kwargs)
            image = output.images[0]

        return image


def init_image_gen_execution_pool(
    model_name: str, num_workers: int, enable_global_rate_limit: bool = True, rate_limit: int = 10, num_gpus: float = 0, gpu_id: str = "0"
):
    """Initialize image generation execution pool.

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
        List of Ray actor handles for image generation workers
    """
    workers = []
    for i in range(num_workers):
        actor_name = f"img-gen-worker-{i}"
        worker = (
            ImageGenExecutionWorker.options(name=actor_name, get_if_exists=True, num_gpus=num_gpus)
            .remote(model_name=model_name, enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, gpu_id=gpu_id)
        )
        workers.append(worker)
        logger.info(f"Created/reused image generation worker {i+1}/{num_workers} (name={actor_name}, gpu={gpu_id})")

    return workers


class ImageGenerationTool(BaseTool):
    """A tool for generating images from text prompts using FLUX.2 Klein model.

    This tool provides image generation functionality with rate limiting and concurrent
    execution support through Ray. It uses the FLUX.2 Klein diffusion model and supports
    multiple GPU workers for parallel processing.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image generation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageGenerationTool
            config:
              type: native
              model_name: "/nfsdata4/zhuhairui/FLUX.2-klein-4B"
              num_workers: 1           # Number of parallel workers
              num_gpus_per_worker: 0.5 # GPU allocation per worker
              rate_limit: 10           # Max concurrent requests
              enable_global_rate_limit: true
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageGenerationTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "ImageGeneration",
                    "description": "Generates a high-quality image from scratch based on a descriptive text prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A detailed, descriptive prompt specifying style, composition, and subject."
                            },
                            "num_inference_steps": {
                                "type": "integer",
                                "description": "Number of denoising steps (default: 28)"
                            },
                            "guidance_scale": {
                                "type": "number",
                                "description": "How closely to follow the prompt (default: 3.5)"
                            }
                        },
                        "required": ["prompt"]
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
        self.default_height = config.get("default_height", 512)
        self.default_width = config.get("default_width", 512)
        self.rate_limit = config.get("rate_limit", 10)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Initialize execution pool
        self.execution_pool = init_image_gen_execution_pool(
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
            f"Initialized ImageGenerationTool with model={self.model_name}, "
            f"num_workers={self.num_workers}, rate_limit={self.rate_limit}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional creation parameters

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "images": [],  # Store generated images
            "prompts": [],  # Store prompts history
            "reward": 0.0,
        }

        logger.debug(f"Created image generation instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image generation tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing prompt and optional generation settings

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the generated image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Extract parameters (FLUX.2 Klein specific defaults)
        prompt = parameters.get("prompt")
        num_inference_steps = parameters.get("num_inference_steps", 4)  # FLUX.2 default
        guidance_scale = parameters.get("guidance_scale", 3.5)  # FLUX.2 default
        negative_prompt = parameters.get("negative_prompt", "")
        height = parameters.get("height", self.default_height)
        width = parameters.get("width", self.default_width)
        seed = parameters.get("seed", None)

        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            error_msg = "Error: 'prompt' is missing or not a string in parameters."
            logger.error(f"[ImageGenerationTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Select worker using round-robin load balancing
        worker = self.execution_pool[self.worker_index % len(self.execution_pool)]
        self.worker_index += 1

        try:
            # Execute image generation
            logger.info(f"Generating image for prompt: {prompt[:50]}...")
            image = await worker.execute.remote(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                seed=seed,
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["images"].append(image)
            self._instance_dict[instance_id]["prompts"].append(prompt)

            # Prepare response
            response_text = f"Generated image for prompt: {prompt}"

            # Metrics
            metrics = {
                "success": True,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "image_size": f"{width}x{height}",
                "worker_id": self.worker_index - 1,
                "seed": seed,
            }

            logger.info(f"Image generation successful for instance {instance_id}")
            return ToolResponse(image=[image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            logger.error(f"[ImageGenerationTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully generated images).
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
            logger.debug(f"Releasing instance {instance_id} with {num_images} generated images")
            del self._instance_dict[instance_id]
