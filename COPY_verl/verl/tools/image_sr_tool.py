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
import subprocess
import tempfile
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
class SRTokenBucketWorker:
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


@ray.remote  # No GPU needed - ncnn handles GPU internally
class SRExecutionWorker:
    """Worker for executing image super resolution with realesrgan-ncnn-vulkan."""

    def __init__(
        self,
        executable_path: str,
        model_name: str = "realesrgan-x4plus",
        scale: int = 4,
        gpu_id: int = 0,
        enable_global_rate_limit: bool = True,
        rate_limit: int = 20,
    ):
        """Initialize SR worker.

        Args:
            executable_path: Path to the realesrgan-ncnn-vulkan executable
            model_name: Model name (realesrgan-x4plus, realesrgan-x4plus-anime, realesr-animevideov3)
            scale: Upscaling factor (default: 4)
            gpu_id: GPU device ID (-1 for CPU)
            enable_global_rate_limit: Whether to enable rate limiting
            rate_limit: Maximum number of concurrent requests
        """
        self.executable_path = executable_path
        self.model_name = model_name
        self.scale = scale
        self.gpu_id = gpu_id
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self._validated = False

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return SRTokenBucketWorker.options(name="sr-rate-limiter", get_if_exists=True).remote(rate_limit)

    def _validate_executable(self):
        """Validate that the executable exists and is runnable."""
        if self._validated:
            return

        if not os.path.exists(self.executable_path):
            raise FileNotFoundError(f"realesrgan-ncnn-vulkan executable not found: {self.executable_path}")

        if not os.access(self.executable_path, os.X_OK):
            raise PermissionError(f"realesrgan-ncnn-vulkan executable is not executable: {self.executable_path}")

        logger.info(f"Validated realesrgan-ncnn-vulkan executable: {self.executable_path}")
        self._validated = True

    def ping(self):
        """Health check method."""
        return True

    def execute(self, image_path: str, outscale: Optional[float] = None, **kwargs) -> Any:
        """Execute image super resolution with optional rate limiting.

        Args:
            image_path: Path to the input image or PIL Image object
            outscale: Output scale factor (default: same as model scale)
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary containing super-resolved image and metadata
        """
        # Validate executable on first use
        self._validate_executable()

        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return self._super_resolve(image_path, outscale, **kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing SR: {e}")
                    raise
        else:
            return self._super_resolve(image_path, outscale, **kwargs)

    def _super_resolve(self, image_path: str, outscale: Optional[float] = None, **kwargs):
        """Internal method to perform super resolution using realesrgan-ncnn-vulkan.

        Args:
            image_path: Path to the input image or PIL Image object
            outscale: Output scale factor (default: same as model scale)
            **kwargs: Additional parameters

        Returns:
            Dictionary with output PIL Image, original size, and new size
        """
        import numpy as np
        from PIL import Image

        # Use default outscale if not specified
        if outscale is None:
            outscale = self.scale

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.png")
            output_path = os.path.join(tmp_dir, "output.png")

            # Handle different input types
            if isinstance(image_path, Image.Image):
                image_path.save(input_path)
                original_size = image_path.size  # (width, height)
            elif isinstance(image_path, np.ndarray):
                # Assume RGB format for numpy array
                Image.fromarray(image_path).save(input_path)
                original_size = (image_path.shape[1], image_path.shape[0])  # (width, height)
            elif isinstance(image_path, str):
                # Use original file path directly, read to get size
                img = Image.open(image_path)
                original_size = img.size  # (width, height)
                input_path = image_path
            else:
                raise ValueError(f"Unsupported image type: {type(image_path)}")

            # Build command
            cmd = [
                self.executable_path,
                '-i', input_path,
                '-o', output_path,
                '-s', str(int(outscale)),
                '-n', self.model_name,
                '-g', str(self.gpu_id),
            ]

            logger.debug(f"Running SR command: {' '.join(cmd)}")

            # Execute realesrgan-ncnn-vulkan (with timeout to prevent hang when Vulkan driver is missing)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "realesrgan-ncnn-vulkan timed out after 120s. "
                    "This usually means NVIDIA Vulkan ICD is not installed. "
                    "Install it with: apt install nvidia-utils-XXX (matching your driver version)"
                )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"realesrgan-ncnn-vulkan failed: {error_msg}")

            # Read output image
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output image not created: {output_path}")

            output_img = Image.open(output_path)
            # Convert to RGB and make a copy to ensure the image is loaded before temp dir is deleted
            output_img = output_img.convert('RGB').copy()
            output_size = output_img.size  # (width, height)

            result = {
                "image": output_img,  # Return PIL Image directly
                "original_size": list(original_size),
                "output_size": list(output_size),
                "scale": outscale,
            }

            return result


def init_sr_execution_pool(
    executable_path: str,
    num_workers: int,
    model_name: str = "realesrgan-x4plus",
    scale: int = 4,
    gpu_id: int = 0,
    enable_global_rate_limit: bool = True,
    rate_limit: int = 20,
):
    """Initialize SR execution pool.

    Args:
        executable_path: Path to the realesrgan-ncnn-vulkan executable
        num_workers: Number of parallel workers
        model_name: Model name (realesrgan-x4plus, realesrgan-x4plus-anime, realesr-animevideov3)
        scale: Upscaling factor (default: 4)
        gpu_id: GPU device ID (-1 for CPU)
        enable_global_rate_limit: Whether to enable rate limiting
        rate_limit: Maximum number of concurrent requests

    Returns:
        List of Ray actor handles for SR workers
    """
    workers = []
    for i in range(num_workers):
        actor_name = f"sr-ncnn-worker-{i}"
        worker = (
            SRExecutionWorker.options(name=actor_name, get_if_exists=True)
            .remote(
                executable_path=executable_path,
                model_name=model_name,
                scale=scale,
                gpu_id=gpu_id,
                enable_global_rate_limit=enable_global_rate_limit,
                rate_limit=rate_limit,
            )
        )
        workers.append(worker)
        logger.info(f"Created/reused ncnn SR worker {i+1}/{num_workers} (name={actor_name})")

    return workers


class ImageSRTool(BaseTool):
    """A tool for upscaling images using realesrgan-ncnn-vulkan.

    This tool provides image super resolution functionality with rate limiting and concurrent
    execution support through Ray. It upscales images using realesrgan-ncnn-vulkan and returns
    high-resolution output.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image super resolution
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageSRTool
            config:
              type: native
              executable_path: "verl/tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan"
              model_name: "realesrgan-x4plus"
              scale: 4
              num_workers: 2           # Number of parallel workers
              gpu_id: 0                # GPU device ID (-1 for CPU)
              rate_limit: 20           # Max concurrent requests
              enable_global_rate_limit: true
            tool_schema:
              type: function
              function:
                name: super_resolve
                description: Upscales an image using Real-ESRGAN (4x)
                parameters:
                  type: object
                  properties: {}
                  required: []
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageSRTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and executable configuration
        self.executable_path = config.get(
            "executable_path",
            os.path.join(os.path.dirname(__file__), "realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan")
        )
        self.model_name = config.get("model_name", "realesrgan-x4plus")
        self.scale = config.get("scale", 4)
        self.gpu_id = config.get("gpu_id", 0)
        self.num_workers = config.get("num_workers", 3)
        self.rate_limit = config.get("rate_limit", 20)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # Initialize execution pool
        self.execution_pool = init_sr_execution_pool(
            executable_path=self.executable_path,
            num_workers=self.num_workers,
            model_name=self.model_name,
            scale=self.scale,
            gpu_id=self.gpu_id,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )

        # Worker index for round-robin load balancing
        self.worker_index = 0

        logger.info(
            f"Initialized ImageSRTool with executable={self.executable_path}, "
            f"model={self.model_name}, scale={self.scale}x, num_workers={self.num_workers}"
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
            "results": [],  # Store SR results
            "images": [],  # Store processed images
            "reward": 0.0,
        }

        logger.debug(f"Created SR instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image super resolution tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the super-resolved image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        from PIL import Image

        # Extract parameters
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

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageSRTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            error_msg = "Error: 'image' must be a PIL Image."
            logger.error(f"[ImageSRTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Select worker using round-robin load balancing
        worker = self.execution_pool[self.worker_index % len(self.execution_pool)]
        self.worker_index += 1

        try:
            # Execute super resolution
            logger.info(f"Upscaling image with scale={self.scale}x")
            result = await worker.execute.remote(
                image_path=image,
            )

            # Parse results
            sr_image = result.get("image")  # PIL Image
            original_size = result.get("original_size", [])
            output_size = result.get("output_size", [])
            scale = result.get("scale", self.scale)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["results"].append(result)
            self._instance_dict[instance_id]["images"].append(sr_image)

            # Format response
            response_text = (
                f"Super resolution completed successfully.\n"
                f"  - Original size: {original_size[0]}x{original_size[1]}\n"
                f"  - Output size: {output_size[0]}x{output_size[1]}\n"
                f"  - Scale factor: {scale}x"
            )

            # Metrics
            metrics = {
                "success": True,
                "original_size": original_size,
                "output_size": output_size,
                "scale": scale,
                "worker_id": self.worker_index - 1,
            }

            logger.info(f"SR successful for instance {instance_id}: {original_size} -> {output_size}")
            return ToolResponse(image=[sr_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Super resolution failed: {e}"
            logger.error(f"[ImageSRTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successful SR operations).
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
            logger.debug(f"Releasing instance {instance_id} with {num_results} SR results")
            del self._instance_dict[instance_id]

    def get_image_from_instance(self, instance_id: str, index: int = -1) -> Optional[Any]:
        """Helper method to retrieve upscaled image from instance history.

        Args:
            instance_id: The instance id of the tool.
            index: Index of the result to retrieve (-1 for latest)

        Returns:
            The upscaled PIL Image, or None if not found
        """
        if instance_id not in self._instance_dict:
            return None

        images = self._instance_dict[instance_id]["images"]
        if not images:
            return None

        try:
            return images[index]
        except IndexError:
            pass

        return None
