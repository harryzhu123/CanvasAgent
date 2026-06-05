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
from typing import Any, List, Optional, TypeVar
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
class GroundingTokenBucketWorker:
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
class GroundingExecutionWorker:
    """Worker for executing object grounding with GroundingDINO model."""

    def __init__(self, model_dir: str, enable_global_rate_limit: bool = True, rate_limit: int = 20, gpu_id: str = "0"):
        """Initialize grounding worker.

        Args:
            model_dir: Path to the GroundingDINO model directory
            enable_global_rate_limit: Whether to enable rate limiting
            rate_limit: Maximum number of concurrent requests
            gpu_id: Physical GPU ID to pin this worker to
        """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.model_dir = model_dir
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.pipe = None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return GroundingTokenBucketWorker.options(name="grounding-rate-limiter", get_if_exists=True).remote(rate_limit)

    def _load_model(self):
        """Lazy load the model to avoid loading during initialization."""
        if self.pipe is not None:
            return

        try:
            from modelscope.pipelines import pipeline

            logger.info(f"Loading GroundingDINO model: {self.model_dir}")
            self.pipe = pipeline('grounding-dino-task', model=self.model_dir)
            logger.info(f"GroundingDINO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO model {self.model_dir}: {e}")
            raise

    def ping(self):
        """Health check method."""
        return True

    def execute(self, image_path: str, text_prompt: str, **kwargs) -> Any:
        """Execute object grounding with optional rate limiting.

        Args:
            image_path: Path to the input image
            text_prompt: Text prompt describing objects to locate
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary containing bounding boxes and other detection results
        """
        # Lazy load model on first use
        self._load_model()

        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return self._ground_objects(image_path, text_prompt, **kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing grounding: {e}")
                    raise
        else:
            return self._ground_objects(image_path, text_prompt, **kwargs)

    def _ground_objects(self, image_path: str, text_prompt: str, **kwargs):
        """Internal method to locate objects using GroundingDINO model.
        
        Args:
            image_path: Path to the input image
            text_prompt: Text prompt describing objects to locate (format: "obj1 . obj2 . obj3 .")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with boxes, logits, and phrases
        """
        import numpy as np
        from PIL import Image
        import tempfile
        import os

        # Handle PIL Image input - save to temp file
        if isinstance(image_path, Image.Image):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image_path.save(tmp.name)
                temp_path = tmp.name
            use_temp = True
        else:
            temp_path = image_path
            use_temp = False

        try:
            # Extract detection parameters
            box_threshold = kwargs.get("box_threshold", 0.35)
            text_threshold = kwargs.get("text_threshold", 0.25)

            # Prepare inputs for GroundingDINO
            inputs = {
                "IMAGE_PATH": temp_path,
                "TEXT_PROMPT": text_prompt,
                "BOX_TRESHOLD": box_threshold,  # Note: typo in original API
                "TEXT_TRESHOLD": text_threshold,
            }

            # Run detection
            output = self.pipe(inputs)

            # Parse results - convert tensors to lists
            boxes = output.get("boxes", [])
            logits = output.get("logits", [])
            phrases = output.get("phrases", [])

            # Convert tensors to lists if needed
            if hasattr(boxes, 'tolist'):
                boxes = boxes.tolist()
            if hasattr(logits, 'tolist'):
                logits = logits.tolist()

            # GroundingDINO predict() returns [cx, cy, w, h] normalized to [0,1].
            # Convert to [xmin, ymin, xmax, ymax] scaled to [0, 1000] to match
            # the system prompt and downstream tools (Crop, SAM).
            converted_boxes = []
            for box in boxes:
                cx, cy, w, h = box
                xmin = (cx - w / 2) * 1000
                ymin = (cy - h / 2) * 1000
                xmax = (cx + w / 2) * 1000
                ymax = (cy + h / 2) * 1000
                converted_boxes.append([
                    round(max(0, xmin)),
                    round(max(0, ymin)),
                    round(min(1000, xmax)),
                    round(min(1000, ymax)),
                ])
            boxes = converted_boxes

            result = {
                "boxes": boxes,
                "logits": logits,
                "phrases": phrases,
            }

            return result
        finally:
            # Clean up temp file if created
            if use_temp and os.path.exists(temp_path):
                os.remove(temp_path)


def init_grounding_execution_pool(
    model_dir: str, num_workers: int, enable_global_rate_limit: bool = True, rate_limit: int = 20, num_gpus: float = 0, gpu_id: str = "0"
):
    """Initialize grounding execution pool.

    Uses named Ray actors with get_if_exists=True so that multiple AgentLoopWorkers
    share the same global pool of GPU workers instead of each creating their own.

    Args:
        model_dir: Path to the GroundingDINO model
        num_workers: Number of parallel workers
        enable_global_rate_limit: Whether to enable rate limiting
        rate_limit: Maximum number of concurrent requests
        num_gpus: Ray GPU resource (0 = manually pinned via CUDA_VISIBLE_DEVICES)
        gpu_id: Physical GPU ID to pin workers to

    Returns:
        List of Ray actor handles for grounding workers
    """
    workers = []
    for i in range(num_workers):
        actor_name = f"grounding-worker-{i}"
        worker = (
            GroundingExecutionWorker.options(name=actor_name, get_if_exists=True, num_gpus=num_gpus)
            .remote(model_dir=model_dir, enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, gpu_id=gpu_id)
        )
        workers.append(worker)
        logger.info(f"Created/reused grounding worker {i+1}/{num_workers} (name={actor_name}, gpu={gpu_id})")

    return workers


class ImageGroundingTool(BaseTool):
    """A tool for locating objects in images using GroundingDINO model.

    This tool provides object grounding functionality with rate limiting and concurrent
    execution support through Ray. It locates objects based on text descriptions and
    returns bounding box coordinates.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the object grounding
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageGroundingTool
            config:
              type: native
              model_dir: "/data/zhuhairui/GroundingDINO"
              num_workers: 4           # Number of parallel workers
              num_gpus_per_worker: 0.25 # GPU allocation per worker
              rate_limit: 20           # Max concurrent requests
              enable_global_rate_limit: true
            tool_schema:
              type: function
              function:
                name: grounding
                description: Locates objects in an image based on text description
                parameters:
                  type: object
                  properties:
                    reference_text:
                      type: string
                      description: Text describing objects to locate (e.g., "the white dog . blue ball .")
                    box_threshold:
                      type: number
                      description: Confidence threshold for bounding boxes (default: 0.35)
                    text_threshold:
                      type: number
                      description: Confidence threshold for text matching (default: 0.25)
                  required:
                    - reference_text
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageGroundingTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.model_dir = config.get("model_dir", "/data/zhuhairui/GroundingDINO")
        self.num_workers = config.get("num_workers", 3)
        self.num_gpus_per_worker = config.get("num_gpus_per_worker", 0)
        self.gpu_id = config.get("gpu_id", "0")
        self.rate_limit = config.get("rate_limit", 20)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Initialize execution pool
        self.execution_pool = init_grounding_execution_pool(
            model_dir=self.model_dir,
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            num_gpus=self.num_gpus_per_worker,
            gpu_id=self.gpu_id,
        )

        # Worker index for round-robin load balancing
        self.worker_index = 0

        logger.info(
            f"Initialized ImageGroundingTool with model_dir={self.model_dir}, "
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
            "results": [],  # Store grounding results
            "prompts": [],  # Store text prompts history
            "images": [],  # Store processed images
            "reward": 0.0,
        }

        logger.debug(f"Created grounding instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image grounding tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - reference_text: Text describing objects to locate
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
                - box_threshold: Confidence threshold for bounding boxes (optional)
                - text_threshold: Confidence threshold for text matching (optional)
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing grounding results
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Extract parameters
        reference_text = parameters.get("reference_text")
        image_param = parameters.get("image") or parameters.get("image_ref")  # Support both 'image' and 'image_ref'
        box_threshold = parameters.get("box_threshold", 0.35)
        text_threshold = parameters.get("text_threshold", 0.25)

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

        # Validate reference_text
        if not reference_text or not isinstance(reference_text, str):
            error_msg = "Error: 'reference_text' is missing or not a string in parameters."
            logger.error(f"[ImageGroundingTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageGroundingTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Ensure text_prompt ends with proper format (obj1 . obj2 . obj3 .)
        text_prompt = reference_text.strip()
        if not text_prompt.endswith('.'):
            text_prompt += ' .'

        # Select worker using round-robin load balancing
        worker = self.execution_pool[self.worker_index % len(self.execution_pool)]
        self.worker_index += 1

        try:
            # Execute grounding
            logger.info(f"Grounding objects with prompt: {text_prompt[:50]}...")
            result = await worker.execute.remote(
                image_path=image,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Parse results
            boxes = result.get("boxes", [])
            phrases = result.get("phrases", [])
            logits = result.get("logits", [])

            # Store results in instance dictionary
            self._instance_dict[instance_id]["results"].append(result)
            self._instance_dict[instance_id]["prompts"].append(text_prompt)
            self._instance_dict[instance_id]["images"].append(image)

            # Format bounding boxes for response
            if boxes and len(boxes) > 0:
                # Convert boxes to list format [x1, y1, x2, y2]
                box_list = []
                for i, box in enumerate(boxes):
                    # Ensure box is a list
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    box_info = {
                        "bounding_box": list(box),
                        "phrase": phrases[i] if i < len(phrases) else "",
                        "confidence": float(logits[i]) if i < len(logits) else 0.0,
                    }
                    box_list.append(box_info)
                
                response_text = f"Found {len(boxes)} objects:\n"
                for box_info in box_list:
                    response_text += f"  - '{box_info['phrase']}': {box_info['bounding_box']} (confidence: {box_info['confidence']:.2f})\n"
            else:
                box_list = []
                response_text = f"No objects matching '{reference_text}' were found in the image."

            # Metrics
            metrics = {
                "success": True,
                "num_objects": len(boxes) if boxes else 0,
                "boxes": box_list,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "worker_id": self.worker_index - 1,
            }

            logger.info(f"Grounding successful for instance {instance_id}: found {len(boxes) if boxes else 0} objects")
            return ToolResponse(text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Grounding failed: {e}"
            logger.error(f"[ImageGroundingTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successful grounding operations).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_results = len(self._instance_dict[instance_id]["results"])
        return float(num_results)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_results = len(self._instance_dict[instance_id]["results"])
            logger.debug(f"Releasing instance {instance_id} with {num_results} grounding results")
            del self._instance_dict[instance_id]
