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


class ImageRotateTool(BaseTool):
    """A tool for rotating images by specified angles.

    This tool provides image rotation functionality supporting:
    - Rotation by 45°, 90°, or 180° only
    - Counter-clockwise rotation

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image rotation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageRotateTool
            config:
              type: native
            tool_schema:
              type: function
              function:
                name: rotate_image
                description: Rotates an image by 45, 90, or 180 degrees
                parameters:
                  type: object
                  properties:
                    angle:
                      type: number
                      enum: [45, 90, 180]
                      description: Rotation angle in degrees (45, 90, or 180)
                  required:
                    - angle
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageRotateTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info("Initialized ImageRotateTool")

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
            "images": [],  # Store rotated images
            "angles": [],  # Store rotation angles
            "original_images": [],  # Store original images
            "reward": 0.0,
        }

        logger.debug(f"Created image rotation instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image rotation tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - angle: Rotation angle (45, 90, or 180)
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the rotated image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Supported angles
        SUPPORTED_ANGLES = [45, 90, 180]

        # Extract parameters
        angle = parameters.get("angle")
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

        # Validate angle
        if angle is None:
            error_msg = "Error: 'angle' is missing in parameters."
            logger.error(f"[ImageRotateTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Convert to float and validate
        try:
            angle = float(angle)
        except (ValueError, TypeError):
            error_msg = f"Error: 'angle' must be a number. Received: {angle}"
            logger.error(f"[ImageRotateTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Check if angle is supported
        if angle not in SUPPORTED_ANGLES:
            error_msg = f"Error: 'angle' must be one of {SUPPORTED_ANGLES}. Received: {angle}"
            logger.error(f"[ImageRotateTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageRotateTool] {error_msg} image_param={image_param}, shared_keys={list(shared_tool_outputs.keys())}, image_data_len={len(image_data)}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                error_msg = "Error: 'image' must be a PIL Image."
                logger.error(f"[ImageRotateTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform rotation
            # expand=True for 45°, False for 90° and 180° (no size change needed)
            expand = (angle == 45)
            fill_color = (255, 255, 255)  # White background for 45° rotation
            
            rotated_image = image.rotate(
                angle,
                expand=expand,
                fillcolor=fill_color,
                resample=Image.Resampling.BICUBIC
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["images"].append(rotated_image)
            self._instance_dict[instance_id]["angles"].append(angle)
            self._instance_dict[instance_id]["original_images"].append(image)

            # Prepare response
            response_text = f"Rotated image by {int(angle)} degrees"

            # Metrics
            metrics = {
                "success": True,
                "angle": angle,
                "original_size": image.size,
                "rotated_size": rotated_image.size,
            }

            logger.info(f"Image rotation successful for instance {instance_id}: {int(angle)}°")
            return ToolResponse(image=[rotated_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image rotation failed: {e}"
            logger.error(f"[ImageRotateTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully rotated images).
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
            logger.debug(f"Releasing instance {instance_id} with {num_images} rotated images")
            del self._instance_dict[instance_id]
