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


class ImageExtractTool(BaseTool):
    """A tool for extracting objects from images using masks.

    This tool provides image extraction functionality supporting:
    - Extracting objects using a mask (from SAM or other segmentation tools)
    - Automatically crops to the bounding box of the mask
    - Returns a cropped PNG-style image with transparent background
    - Supports referencing images/masks from shared_tool_outputs

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image extraction
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageExtractTool
            config:
              type: native
            tool_schema:
              type: function
              function:
                name: Extract
                description: Extracts an object from an image using a mask. Returns a Cropped PNG-style image with a transparent background.
                parameters:
                  type: object
                  properties:
                    mask_ref:
                      type: string
                      description: Reference to the mask in shared outputs (e.g., "SAM_0")
                    image_ref:
                      type: string
                      description: Optional reference to the source image (e.g., "Crop_0"). If not provided, uses the initial image.
                  required:
                    - mask_ref
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageExtractTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info("Initialized ImageExtractTool")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional creation parameters. May contain 'image' and/or 'mask' keys,
                or 'create_kwargs' containing these.

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

        # Get image and mask from kwargs (optional - can be provided via shared_tool_outputs)
        image = kwargs.get("image")
        mask = kwargs.get("mask")

        self._instance_dict[instance_id] = {
            "image": image,
            "mask": mask,
            "extracted_images": [],
            "masks_used": [],
            "original_images": [],
            "reward": 0.0,
        }

        logger.debug(f"Created image extract instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image extract tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - mask_ref: Reference to mask in shared_tool_outputs (e.g., "SAM_0")
                - image_ref: Optional reference to image in shared_tool_outputs
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images/masks

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the cropped extracted image with transparent background
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool (includes bbox coordinates)
        """
        # Extract parameters
        # Support both 'mask' (schema name) and 'mask_ref' (legacy name)
        mask_param = parameters.get("mask") or parameters.get("mask_ref")
        # Support both 'image' (schema name) and 'image_ref' (legacy name)
        image_param = parameters.get("image") or parameters.get("image_ref")

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

        # Get mask from various sources (priority order):
        # 1. shared_tool_outputs (if mask parameter is provided)
        # 2. instance_dict (from create_kwargs)
        mask = None
        if mask_param and mask_param in shared_tool_outputs:
            mask = shared_tool_outputs[mask_param]
        elif self._instance_dict[instance_id]["mask"] is not None:
            mask = self._instance_dict[instance_id]["mask"]

        # Validate image
        if image is None:
            error_msg = "Error: No image available. Provide image via create_kwargs, image parameter, or image_data."
            logger.error(f"[ImageExtractTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate mask
        if mask is None:
            error_msg = "Error: No mask available. Provide mask via create_kwargs or mask parameter."
            logger.error(f"[ImageExtractTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                error_msg = "Error: 'image' must be a PIL Image."
                logger.error(f"[ImageExtractTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Ensure mask is PIL Image
            if not isinstance(mask, Image.Image):
                error_msg = "Error: 'mask' must be a PIL Image."
                logger.error(f"[ImageExtractTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Get image and mask dimensions
            img_width, img_height = image.size
            mask_width, mask_height = mask.size

            # Validate dimensions match
            if img_width != mask_width or img_height != mask_height:
                error_msg = f"Error: Image and mask dimensions must match. Image: {image.size}, Mask: {mask.size}"
                logger.error(f"[ImageExtractTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Convert mask to grayscale (L mode)
            if mask.mode != 'L':
                mask_l = mask.convert('L')
            else:
                mask_l = mask.copy()

            # Find bounding box of the mask (non-zero region)
            bbox = mask_l.getbbox()
            if bbox is None:
                error_msg = "Error: Mask is empty (no non-zero pixels found)."
                logger.error(f"[ImageExtractTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            xmin, ymin, xmax, ymax = bbox

            # Crop the image and mask to the bounding box
            cropped_image = image.crop(bbox)
            cropped_mask = mask_l.crop(bbox)

            # Convert cropped image to RGBA
            if cropped_image.mode != 'RGBA':
                cropped_rgba = cropped_image.convert('RGBA')
            else:
                cropped_rgba = cropped_image.copy()

            # Apply cropped mask as alpha channel
            # White (255) in mask = fully opaque, Black (0) = fully transparent
            cropped_rgba.putalpha(cropped_mask)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["extracted_images"].append(cropped_rgba)
            self._instance_dict[instance_id]["masks_used"].append(mask)
            self._instance_dict[instance_id]["original_images"].append(image)

            # Prepare response
            response_text = f"Extracted object from image using mask. Cropped from bbox: [{xmin},{ymin},{xmax},{ymax}], Output size: {cropped_rgba.size}"

            # Metrics
            metrics = {
                "success": True,
                "original_size": image.size,
                "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                "extracted_size": cropped_rgba.size,
                "output_mode": cropped_rgba.mode,
            }

            logger.info(f"Image extraction successful for instance {instance_id}: bbox={bbox}")
            return ToolResponse(image=[cropped_rgba], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image extraction failed: {e}"
            logger.error(f"[ImageExtractTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully extracted images).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_images = len(self._instance_dict[instance_id]["extracted_images"])
        return float(num_images)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_images = len(self._instance_dict[instance_id]["extracted_images"])
            logger.debug(f"Releasing instance {instance_id} with {num_images} extracted images")
            del self._instance_dict[instance_id]
