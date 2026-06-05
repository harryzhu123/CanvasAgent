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
class SAMTokenBucketWorker:
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
class SAMExecutionWorker:
    """Worker for executing image segmentation with SAM model."""

    def __init__(self, model_path: str, model_type: str = "vit_h", enable_global_rate_limit: bool = True, rate_limit: int = 20, gpu_id: str = "0"):
        """Initialize SAM worker.

        Args:
            model_path: Path to the SAM model checkpoint file
            model_type: SAM model type (vit_h, vit_l, vit_b)
            enable_global_rate_limit: Whether to enable rate limiting
            rate_limit: Maximum number of concurrent requests
            gpu_id: Physical GPU ID to pin this worker to
        """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.model_path = model_path
        self.model_type = model_type
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.predictor = None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return SAMTokenBucketWorker.options(name="sam-rate-limiter", get_if_exists=True).remote(rate_limit)

    def _load_model(self):
        """Lazy load the model to avoid loading during initialization."""
        if self.predictor is not None:
            return

        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor

            logger.info(f"Loading SAM model: {self.model_path} (type: {self.model_type})")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
            sam.to(device=device)

            self.predictor = SamPredictor(sam)
            logger.info(f"SAM model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load SAM model {self.model_path}: {e}")
            raise

    def ping(self):
        """Health check method."""
        return True

    def execute(self, image_path: str, bounding_box: List[float], **kwargs) -> Any:
        """Execute image segmentation with optional rate limiting.

        Args:
            image_path: Path to the input image or PIL Image object
            bounding_box: Bounding box coordinates [x_min, y_min, x_max, y_max]
                in normalized 0-1000 space
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary containing binary mask and other segmentation results
        """
        # Lazy load model on first use
        self._load_model()

        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return self._segment(image_path, bounding_box, **kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing SAM segmentation: {e}")
                    raise
        else:
            return self._segment(image_path, bounding_box, **kwargs)

    def _segment(self, image_path: str, bounding_box: List[float], **kwargs):
        """Internal method to segment objects using SAM model.

        Args:
            image_path: Path to the input image or PIL Image object
            bounding_box: Bounding box coordinates [x_min, y_min, x_max, y_max]
                in normalized 0-1000 space
            **kwargs: Additional parameters

        Returns:
            Dictionary with mask, score, and metadata
        """
        import numpy as np
        from PIL import Image
        import tempfile
        import os

        # Handle PIL Image input - convert to numpy array
        if isinstance(image_path, Image.Image):
            image = np.array(image_path.convert("RGB"))
            image_size = image_path.size  # (width, height)
        elif isinstance(image_path, str):
            pil_image = Image.open(image_path).convert("RGB")
            image = np.array(pil_image)
            image_size = pil_image.size
        elif isinstance(image_path, np.ndarray):
            image = image_path
            image_size = (image.shape[1], image.shape[0])  # (width, height)
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")

        # Set image for predictor
        self.predictor.set_image(image)

        # Convert bounding box to numpy array
        box = np.array(bounding_box)

        # Extract optional parameters
        multimask_output = kwargs.get("multimask_output", False)

        # Run prediction with bounding box prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=multimask_output,
        )

        # Get the best mask (highest score)
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0]

        # Convert mask to binary (0/1) format
        binary_mask = mask.astype(np.uint8)

        result = {
            "mask": binary_mask.tolist(),  # Convert to list for serialization
            "score": float(score),
            "mask_shape": list(binary_mask.shape),
            "image_size": list(image_size),
            "bounding_box": list(bounding_box),
        }

        return result


def init_sam_execution_pool(
    model_path: str,
    num_workers: int,
    model_type: str = "vit_h",
    enable_global_rate_limit: bool = True,
    rate_limit: int = 20,
    num_gpus: float = 0,
    gpu_id: str = "0",
):
    """Initialize SAM execution pool.

    Uses named Ray actors with get_if_exists=True so that multiple AgentLoopWorkers
    share the same global pool of GPU workers instead of each creating their own.

    Args:
        model_path: Path to the SAM model checkpoint
        num_workers: Number of parallel workers
        model_type: SAM model type (vit_h, vit_l, vit_b)
        enable_global_rate_limit: Whether to enable rate limiting
        rate_limit: Maximum number of concurrent requests
        num_gpus: Ray GPU resource (0 = manually pinned via CUDA_VISIBLE_DEVICES)
        gpu_id: Physical GPU ID to pin workers to

    Returns:
        List of Ray actor handles for SAM workers
    """
    workers = []
    for i in range(num_workers):
        actor_name = f"sam-worker-{i}"
        worker = (
            SAMExecutionWorker.options(name=actor_name, get_if_exists=True, num_gpus=num_gpus)
            .remote(
                model_path=model_path,
                model_type=model_type,
                enable_global_rate_limit=enable_global_rate_limit,
                rate_limit=rate_limit,
                gpu_id=gpu_id,
            )
        )
        workers.append(worker)
        logger.info(f"Created/reused SAM worker {i+1}/{num_workers} (name={actor_name}, gpu={gpu_id})")

    return workers


class ImageSAMTool(BaseTool):
    """A tool for segmenting objects in images using SAM (Segment Anything Model).

    This tool provides image segmentation functionality with rate limiting and concurrent
    execution support through Ray. It segments objects based on bounding box prompts and
    returns binary masks.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image segmentation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageSAMTool
            config:
              type: native
              model_path: "/path/to/sam_vit_h.pth"
              model_type: "vit_h"
              num_workers: 4           # Number of parallel workers
              num_gpus_per_worker: 0.5 # GPU allocation per worker
              rate_limit: 20           # Max concurrent requests
              enable_global_rate_limit: true
            tool_schema:
              type: function
              function:
                name: segment
                description: Segments objects in an image using SAM with bounding box prompt
                parameters:
                  type: object
                  properties:
                    bounding_box:
                      type: array
                      description: Bounding box [x_min, y_min, x_max, y_max] in normalized 0-1000 space
                  required:
                    - bounding_box
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageSAMTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.model_path = config.get("model_path", "/data/zhuhairui/sam_vit_h.pth")
        self.model_type = config.get("model_type", "vit_h")
        self.num_workers = config.get("num_workers", 3)
        self.num_gpus_per_worker = config.get("num_gpus_per_worker", 0)
        self.gpu_id = config.get("gpu_id", "0")
        self.rate_limit = config.get("rate_limit", 20)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Initialize execution pool
        self.execution_pool = init_sam_execution_pool(
            model_path=self.model_path,
            num_workers=self.num_workers,
            model_type=self.model_type,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            num_gpus=self.num_gpus_per_worker,
            gpu_id=self.gpu_id,
        )

        # Worker index for round-robin load balancing
        self.worker_index = 0

        logger.info(
            f"Initialized ImageSAMTool with model_path={self.model_path}, "
            f"model_type={self.model_type}, num_workers={self.num_workers}, rate_limit={self.rate_limit}"
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
            "results": [],  # Store segmentation results
            "boxes": [],  # Store bounding boxes history
            "images": [],  # Store processed images
            "reward": 0.0,
        }

        logger.debug(f"Created SAM instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image segmentation tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - bounding_box: Bounding box [x_min, y_min, x_max, y_max]
                  in normalized 0-1000 space
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Crop_0")
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the mask as PIL Image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        from PIL import Image
        import numpy as np

        # Extract parameters
        # Support both 'bbox' (schema name) and 'bounding_box' (legacy name)
        bbox_param = parameters.get("bbox") or parameters.get("bounding_box")
        image_param = parameters.get("image") or parameters.get("image_ref")  # Support both 'image' and 'image_ref'

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

        # Validate bbox parameter
        if bbox_param is None:
            error_msg = "Error: 'bbox' is missing in parameters."
            logger.error(f"[ImageSAMTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Parse bbox - support both string "[xmin,ymin,xmax,ymax]" and list/tuple formats
        if isinstance(bbox_param, str):
            try:
                # Handle "[xmin,ymin,xmax,ymax]" or "xmin,ymin,xmax,ymax" formats
                bbox_str = bbox_param.strip().strip("[]")
                coords = [float(c.strip()) for c in bbox_str.split(",")]
                if len(coords) != 4:
                    raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
                bounding_box = coords
            except (ValueError, TypeError, AttributeError) as e:
                error_msg = f"Error: 'bbox' must be in format '[xmin,ymin,xmax,ymax]'. Received: {bbox_param}"
                logger.error(f"[ImageSAMTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}
        elif isinstance(bbox_param, (list, tuple)):
            if len(bbox_param) != 4:
                error_msg = f"Error: 'bbox' must have 4 coordinates [x_min, y_min, x_max, y_max]. Received: {bbox_param}"
                logger.error(f"[ImageSAMTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}
            bounding_box = list(bbox_param)
        else:
            error_msg = f"Error: 'bbox' must be a string or list of 4 numbers. Received: {bbox_param}"
            logger.error(f"[ImageSAMTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageSAMTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Convert from 0-1000 normalized coordinates to absolute pixels.
        # All bbox coordinates in the pipeline use the 0-1000 range
        # (consistent with Grounding tool output).
        # We need image dimensions for conversion.
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            img_w, img_h = image.size
        else:
            # Fallback: pass as-is if we can't determine size
            img_w, img_h = 1000, 1000
        bounding_box = [
            bounding_box[0] * img_w / 1000,
            bounding_box[1] * img_h / 1000,
            bounding_box[2] * img_w / 1000,
            bounding_box[3] * img_h / 1000,
        ]

        # Select worker using round-robin load balancing
        worker = self.execution_pool[self.worker_index % len(self.execution_pool)]
        self.worker_index += 1

        try:
            # Execute segmentation
            logger.info(f"Segmenting with bounding box: {bounding_box}")
            result = await worker.execute.remote(
                image_path=image,
                bounding_box=bounding_box,
            )

            # Parse results
            mask_shape = result.get("mask_shape", [])
            score = result.get("score", 0.0)
            image_size = result.get("image_size", [])

            # Store results in instance dictionary
            self._instance_dict[instance_id]["results"].append(result)
            self._instance_dict[instance_id]["boxes"].append(bounding_box)
            self._instance_dict[instance_id]["images"].append(image)

            # Calculate mask statistics and convert to PIL Image
            mask = result.get("mask", [])
            mask_image = None
            if mask:
                mask_array = np.array(mask, dtype=np.uint8)
                mask_area = int(np.sum(mask_array))
                total_area = mask_array.size
                mask_ratio = mask_area / total_area if total_area > 0 else 0.0

                # Convert binary mask (0/1) to grayscale PIL Image (0/255)
                # White (255) = foreground, Black (0) = background
                mask_image = Image.fromarray(mask_array * 255, mode='L')
            else:
                mask_area = 0
                mask_ratio = 0.0

            # Format response
            response_text = (
                f"Segmentation completed successfully.\n"
                f"  - Mask shape: {mask_shape}\n"
                f"  - Confidence score: {score:.4f}\n"
                f"  - Mask area: {mask_area} pixels ({mask_ratio:.2%} of image)\n"
                f"  - Bounding box: {bounding_box}"
            )

            # Metrics
            metrics = {
                "success": True,
                "mask_shape": mask_shape,
                "score": score,
                "mask_area": mask_area,
                "mask_ratio": mask_ratio,
                "image_size": image_size,
                "bounding_box": bounding_box,
                "worker_id": self.worker_index - 1,
            }

            logger.info(f"Segmentation successful for instance {instance_id}: score={score:.4f}")
            # Return mask as PIL Image so it can be stored in shared_tool_outputs
            return ToolResponse(image=[mask_image] if mask_image else None, text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Segmentation failed: {e}"
            logger.error(f"[ImageSAMTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successful segmentation operations).
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
            logger.debug(f"Releasing instance {instance_id} with {num_results} segmentation results")
            del self._instance_dict[instance_id]

    def get_mask_from_instance(self, instance_id: str, index: int = -1) -> Optional[Any]:
        """Helper method to retrieve mask from instance history.

        Args:
            instance_id: The instance id of the tool.
            index: Index of the result to retrieve (-1 for latest)

        Returns:
            The binary mask as numpy array, or None if not found
        """
        if instance_id not in self._instance_dict:
            return None

        results = self._instance_dict[instance_id]["results"]
        if not results:
            return None

        try:
            import numpy as np
            result = results[index]
            mask = result.get("mask")
            if mask is not None:
                return np.array(mask, dtype=np.uint8)
        except (IndexError, KeyError):
            pass

        return None
