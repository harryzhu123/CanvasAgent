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

from PIL import Image, ImageDraw, ImageFont

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageOverlayerTool(BaseTool):
    """A tool for compositing text or objects onto background images.

    This tool provides image overlay functionality supporting:
    - Overlaying text onto images (watermarks, labels)
    - Overlaying extracted objects onto background images (collages)
    - Positioning overlays using normalized coordinates
    - Supports referencing images from shared_tool_outputs

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image overlay
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: ImageOverlayerTool
            config:
              type: native
            tool_schema:
              type: function
              function:
                name: Overlayer
                description: Composites text or an extracted object onto a background image. Useful for adding watermarks, labels, or creating collages.
                parameters:
                  type: object
                  properties:
                    overlay_type:
                      type: string
                      enum: [text, object]
                      description: Specify whether you are overlaying text or an image object.
                    content:
                      type: string
                      description: For text mode, the text string to add. For object mode, the reference to the overlay image (e.g., "Extract_0").
                    position:
                      type: string
                      description: Normalized coordinates [x, y] for the center of the overlay. Defaults to [500, 500] (center).
                    base_image_ref:
                      type: string
                      description: Optional reference to background image (e.g., "Crop_0"). If not provided, uses the initial image.
                  required:
                    - overlay_type
                    - content
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize ImageOverlayerTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        logger.info("Initialized ImageOverlayerTool")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional creation parameters. May contain 'base_image' key,
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

        # Get base_image from kwargs (optional - can be provided via shared_tool_outputs)
        base_image = kwargs.get("base_image") or kwargs.get("image")

        self._instance_dict[instance_id] = {
            "base_image": base_image,
            "result_images": [],
            "overlays": [],
            "reward": 0.0,
        }

        logger.debug(f"Created image overlayer instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the image overlayer tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing:
                - overlay_type: "text" or "object"
                - content: Text string (for text mode) or reference to overlay image (for object mode, e.g., "Extract_0")
                - position: Optional position string "[x, y]" for center of overlay (normalized 0-1000)
                - base_image_ref: Optional reference to background image in shared_tool_outputs
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing the composited image
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool
        """
        # Extract parameters
        overlay_type = parameters.get("overlay_type")
        content = parameters.get("content")
        position = parameters.get("position")
        # Support both 'base_image' (schema name) and 'base_image_ref' (legacy name)
        base_image_param = parameters.get("base_image") or parameters.get("base_image_ref")

        # Get shared data
        shared_tool_outputs = kwargs.get("shared_tool_outputs", {})
        image_data = kwargs.get("image_data") or []

        # Get base_image from various sources (priority order):
        # 1. shared_tool_outputs (if base_image parameter is provided)
        # 2. instance_dict (from create_kwargs)
        # 3. image_data[0] (first image in the list)
        base_image = None
        if base_image_param and base_image_param in shared_tool_outputs:
            base_image = shared_tool_outputs[base_image_param]
        elif self._instance_dict[instance_id]["base_image"] is not None:
            base_image = self._instance_dict[instance_id]["base_image"]
        elif image_data:
            base_image = image_data[0]

        # Validate base_image
        if base_image is None:
            error_msg = "Error: No base image available. Provide via create_kwargs, base_image parameter, or image_data."
            logger.error(f"[ImageOverlayerTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate overlay_type parameter
        if overlay_type is None:
            error_msg = "Error: 'overlay_type' is missing in parameters."
            logger.error(f"[ImageOverlayerTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        if overlay_type not in ["text", "object"]:
            error_msg = f"Error: 'overlay_type' must be 'text' or 'object'. Received: {overlay_type}"
            logger.error(f"[ImageOverlayerTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Validate content parameter
        if content is None:
            error_msg = "Error: 'content' is missing in parameters."
            logger.error(f"[ImageOverlayerTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        # Parse position parameter (default to center [500, 500] normalized)
        try:
            if position is None:
                pos_x, pos_y = 500, 500
            elif isinstance(position, (list, tuple)):
                # Model may output position as JSON array [x, y] instead of string
                if len(position) != 2:
                    raise ValueError(f"Expected 2 coordinates, got {len(position)}")
                pos_x, pos_y = float(position[0]), float(position[1])
            else:
                # Handle both "[x, y]" and "x, y" string formats
                pos_str = str(position).strip().strip("[]")
                coords = [float(c.strip()) for c in pos_str.split(",")]
                if len(coords) != 2:
                    raise ValueError(f"Expected 2 coordinates, got {len(coords)}")
                pos_x, pos_y = coords
        except (ValueError, TypeError, AttributeError) as e:
            error_msg = f"Error: 'position' must be in format '[x, y]'. Received: {position}"
            logger.error(f"[ImageOverlayerTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Ensure base_image is PIL Image
            if not isinstance(base_image, Image.Image):
                error_msg = "Error: 'base_image' must be a PIL Image."
                logger.error(f"[ImageOverlayerTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Get base image dimensions
            base_width, base_height = base_image.size

            # Convert normalized position to pixel coordinates
            # Assuming normalized coordinates are in range [0, 1000]
            pixel_x = int(pos_x / 1000 * base_width)
            pixel_y = int(pos_y / 1000 * base_height)

            # Convert base image to RGBA for compositing
            if base_image.mode != 'RGBA':
                result_image = base_image.convert('RGBA')
            else:
                result_image = base_image.copy()

            if overlay_type == "text":
                # Text overlay mode
                if not isinstance(content, str):
                    error_msg = "Error: 'content' must be a string for text overlay."
                    logger.error(f"[ImageOverlayerTool] {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                # Create drawing context
                draw = ImageDraw.Draw(result_image)

                # Try to load a font, fall back to default if not available
                font_size = max(20, min(base_width, base_height) // 20)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except (IOError, OSError):
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
                    except (IOError, OSError):
                        font = ImageFont.load_default()

                # Get text bounding box
                text_bbox = draw.textbbox((0, 0), content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Calculate top-left position (centered on pixel_x, pixel_y)
                text_x = pixel_x - text_width // 2
                text_y = pixel_y - text_height // 2

                # Draw text with black color
                draw.text((text_x, text_y), content, font=font, fill=(0, 0, 0, 255))

                overlay_info = {
                    "type": "text",
                    "content": content,
                    "position": (pixel_x, pixel_y),
                    "text_size": (text_width, text_height),
                }

            else:  # overlay_type == "object"
                # Object overlay mode - content should be a reference to an image in shared_tool_outputs
                overlay_image = None

                # Try to get overlay image from shared_tool_outputs using content as reference
                if isinstance(content, str) and content in shared_tool_outputs:
                    overlay_image = shared_tool_outputs[content]
                elif isinstance(content, Image.Image):
                    # Direct PIL Image (for backward compatibility)
                    overlay_image = content

                if overlay_image is None:
                    error_msg = f"Error: Could not find overlay image. 'content' should be a reference like 'Extract_0'. Received: {content}"
                    logger.error(f"[ImageOverlayerTool] {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                if not isinstance(overlay_image, Image.Image):
                    error_msg = "Error: Overlay image must be a PIL Image."
                    logger.error(f"[ImageOverlayerTool] {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                # Object overlays must already carry an alpha channel. Auto-converting
                # RGB to RGBA would make every pixel fully opaque and paste a rectangle.
                if overlay_image.mode != 'RGBA':
                    error_msg = (
                        "Error: For overlay_type='object', the overlay image must be an RGBA image with "
                        "a transparent background, typically produced by the Extract tool (e.g., 'Extract_0'). "
                        f"Received mode: {overlay_image.mode}. Do not pass a regular RGB image as the object overlay."
                    )
                    logger.error(f"[ImageOverlayerTool] {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                # Get overlay dimensions
                overlay_width, overlay_height = overlay_image.size

                # Calculate top-left position (centered on pixel_x, pixel_y)
                paste_x = pixel_x - overlay_width // 2
                paste_y = pixel_y - overlay_height // 2

                # Paste overlay onto result image using alpha compositing
                result_image.paste(overlay_image, (paste_x, paste_y), overlay_image)

                overlay_info = {
                    "type": "object",
                    "content_ref": content if isinstance(content, str) else "direct_image",
                    "overlay_size": overlay_image.size,
                    "position": (pixel_x, pixel_y),
                    "paste_position": (paste_x, paste_y),
                }

            # Convert back to RGB if original was RGB
            if base_image.mode == 'RGB':
                result_image = result_image.convert('RGB')

            # Store results in instance dictionary
            self._instance_dict[instance_id]["result_images"].append(result_image)
            self._instance_dict[instance_id]["overlays"].append(overlay_info)

            # Prepare response
            response_text = f"Overlaid {overlay_type} onto image at position [{pos_x},{pos_y}]. Output size: {result_image.size}"

            # Metrics
            metrics = {
                "success": True,
                "base_size": base_image.size,
                "output_size": result_image.size,
                "overlay_type": overlay_type,
                "normalized_position": (pos_x, pos_y),
                "pixel_position": (pixel_x, pixel_y),
            }

            logger.info(f"Image overlay successful for instance {instance_id}: {overlay_type} at ({pixel_x}, {pixel_y})")
            return ToolResponse(image=[result_image], text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"Image overlay failed: {e}"
            logger.error(f"[ImageOverlayerTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully composited images).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_images = len(self._instance_dict[instance_id]["result_images"])
        return float(num_images)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_images = len(self._instance_dict[instance_id]["result_images"])
            logger.debug(f"Releasing instance {instance_id} with {num_images} composited images")
            del self._instance_dict[instance_id]
