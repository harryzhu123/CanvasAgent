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
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageFlipTool(BaseTool):
    """A tool for flipping images horizontally (left-right mirror).

    This tool provides image flipping functionality supporting:
    - Horizontal flip (left-right mirror)

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image flip
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageFlipTool
            config:
              type: native
            tool_schema:
              type: function
              function:
                name: flip_image
                description: Flips an image horizontally (left-right mirror)
                parameters:
                  type: object
                  properties: {}
                  required: []
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageFlipTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info("Initialized ImageFlipTool")

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
            "images": [],  # Store flipped images
            "directions": [],  # Store flip directions
            "original_images": [],  # Store original images
            "reward": 0.0,
        }

        logger.debug(f"Created image flip instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image flip tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the flipped image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
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
            logger.error(f"[ImageFlipTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                error_msg = "Error: 'image' must be a PIL Image."
                logger.error(f"[ImageFlipTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform horizontal flip
            flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["images"].append(flipped_image)
            self._instance_dict[instance_id]["directions"].append("horizontal")
            self._instance_dict[instance_id]["original_images"].append(image)

            # Prepare response
            response_text = "Flipped image horizontally"

            # Metrics
            metrics = {
                "success": True,
                "direction": "horizontal",
                "image_size": image.size,
            }

            logger.info(f"Image flip successful for instance {instance_id}: horizontal")
            return ToolResponse(image=[flipped_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image flip failed: {e}"
            logger.error(f"[ImageFlipTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully flipped images).
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
            logger.debug(f"Releasing instance {instance_id} with {num_images} flipped images")
            del self._instance_dict[instance_id]
