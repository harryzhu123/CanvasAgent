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


class ImageCropTool(BaseTool):
    """A tool for cropping images using bounding box coordinates.

    This tool provides image cropping functionality supporting:
    - Cropping by bounding box [xmin, ymin, xmax, ymax] in normalized 0-1000 space
    - Validation of bounding box coordinates within image boundaries
    - Image passed via create_kwargs or shared_tool_outputs

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image cropping
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageCropTool
            config:
              type: native
            tool_schema:
              type: function
              function:
                name: Crop
                description: According to a bounding box, create a cropped image
                parameters:
                  type: object
                  properties:
                    bbox:
                      type: string
                      description: normalized coordinates [xmin,ymin,xmax,ymax] in 0-1000 space
                    image_ref:
                      type: string
                      description: Optional reference to image (e.g., "Extract_0"). If not provided, uses the initial image.
                  required:
                    - bbox
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageCropTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info("Initialized ImageCropTool")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional creation parameters. Should contain 'image' key with image data,
                or 'create_kwargs' containing {'image': image_data}.

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

        # Get image from kwargs (optional - can also be provided via shared_tool_outputs in execute)
        image = kwargs.get("image")

        self._instance_dict[instance_id] = {
            "image": image,  # May be None if image will be provided via shared_tool_outputs
            "cropped_images": [],
            "bounding_boxes": [],
            "reward": 0.0,
        }

        logger.debug(f"Created image crop instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image crop tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - bbox: Bounding box coordinates as string "[xmin,ymin,xmax,ymax]"
                  in normalized 0-1000 space
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the cropped image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Extract parameters
        bbox = parameters.get("bbox")
        image_param = parameters.get("image") or parameters.get("image_ref")  # Support both 'image' and 'image_ref'

        # Get image from various sources (priority order):
        # 1. shared_tool_outputs (if image parameter is provided)
        # 2. instance_dict (from create_kwargs)
        # 3. image_data[0] (first image in the list)
        image = None
        shared_tool_outputs = kwargs.get("shared_tool_outputs", {})
        image_data = kwargs.get("image_data") or []

        if image_param and image_param in shared_tool_outputs:
            image = shared_tool_outputs[image_param]
        elif self._instance_dict[instance_id]["image"] is not None:
            image = self._instance_dict[instance_id]["image"]
        elif image_data:
            image = image_data[0]  # Use first image as default

        # Validate bbox parameter
        if bbox is None:
            error_msg = "Error: 'bbox' is missing in parameters."
            logger.error(f"[ImageCropTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Parse bbox to coordinates — accept both string and list/tuple formats
        try:
            if isinstance(bbox, (list, tuple)):
                coords = [int(float(c)) for c in bbox]
            else:
                # Handle "[xmin,ymin,xmax,ymax]" and "xmin,ymin,xmax,ymax" string formats
                bbox_str = bbox.strip().strip("[]")
                coords = [int(float(c.strip())) for c in bbox_str.split(",")]
            if len(coords) != 4:
                raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
            xmin, ymin, xmax, ymax = coords
        except (ValueError, TypeError, AttributeError) as e:
            error_msg = f"Error: 'bbox' must be in format '[xmin,ymin,xmax,ymax]'. Received: {bbox}"
            logger.error(f"[ImageCropTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate non-negative values and ordering
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            error_msg = f"Error: All coordinates must be non-negative. Received: [{xmin},{ymin},{xmax},{ymax}]"
            logger.error(f"[ImageCropTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        if xmax <= xmin or ymax <= ymin:
            error_msg = f"Error: xmax must be > xmin and ymax must be > ymin. Received: [{xmin},{ymin},{xmax},{ymax}]"
            logger.error(f"[ImageCropTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image_ref, or image_data."
            logger.error(f"[ImageCropTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                error_msg = "Error: 'image' must be a PIL Image."
                logger.error(f"[ImageCropTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Get image dimensions
            img_width, img_height = image.size

            # Convert from 0-1000 normalized coordinates to absolute pixels.
            # All bbox coordinates in the pipeline use the 0-1000 range
            # (consistent with Grounding tool output).
            xmin = int(xmin * img_width / 1000)
            ymin = int(ymin * img_height / 1000)
            xmax = int(xmax * img_width / 1000)
            ymax = int(ymax * img_height / 1000)

            # Clamp to image boundaries
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(0, min(xmax, img_width))
            ymax = max(0, min(ymax, img_height))

            # Validate after conversion
            if xmax <= xmin or ymax <= ymin:
                error_msg = f"Error: Invalid crop region after coordinate conversion. [{xmin},{ymin},{xmax},{ymax}]"
                logger.error(f"[ImageCropTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Crop box (left, upper, right, lower)
            crop_box = (xmin, ymin, xmax, ymax)

            # Perform cropping
            cropped_image = image.crop(crop_box)

            # Store results in instance dictionary
            bounding_box = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            self._instance_dict[instance_id]["cropped_images"].append(cropped_image)
            self._instance_dict[instance_id]["bounding_boxes"].append(bounding_box)

            # Prepare response
            response_text = f"Cropped image with bounding box: [{xmin},{ymin},{xmax},{ymax}]"

            # Metrics
            metrics = {
                "success": True,
                "bounding_box": bounding_box,
                "original_size": image.size,
                "cropped_size": cropped_image.size,
            }

            logger.info(f"Image crop successful for instance {instance_id}: {bounding_box}")
            return ToolResponse(image=[cropped_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image cropping failed: {e}"
            logger.error(f"[ImageCropTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully cropped images).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_images = len(self._instance_dict[instance_id]["cropped_images"])
        return float(num_images)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_images = len(self._instance_dict[instance_id]["cropped_images"])
            logger.debug(f"Releasing instance {instance_id} with {num_images} cropped images")
            del self._instance_dict[instance_id]
