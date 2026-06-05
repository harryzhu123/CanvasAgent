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

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OCRTool(BaseTool):
    """A tool for performing OCR (Optical Character Recognition) on images.

    This tool uses RapidOCR to extract text from images and returns
    the recognized text with bounding boxes and confidence scores.

    Output format:
        - When use_det=True: Returns dict with "box" (in [xmin, ymin, xmax, ymax] format), "text", and "confidence"
        - When use_det=False: Returns dict with "text" and "confidence" only
        - Box coordinates are absolute pixel coordinates from the top-left corner (0,0)

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute OCR on the image
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance

    Example configuration:
        tools:
          - class_name: OCRTool
            config:
              type: native
              use_det: true    # Text detection (default: true)
              use_cls: false   # Text classification/orientation (default: false)
              use_rec: false   # Text recognition (default: false)
            tool_schema:
              type: function
              function:
                name: ocr
                description: |
                  Performs OCR on an image and returns recognized text with bounding boxes and confidence scores.
                  When use_det=True, returns box coordinates in [xmin, ymin, xmax, ymax] format (absolute pixel coordinates).
                parameters:
                  type: object
                  properties:
                    image:
                      type: string
                      description: The image file path or image data to perform OCR on
                  required: [image]
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize OCRTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Default OCR settings from config
        self._default_use_det = config.get("use_det", True)
        self._default_use_cls = config.get("use_cls", False)
        self._default_use_rec = config.get("use_rec", True)

        # Initialize RapidOCR engine
        self._engine = RapidOCR()

        logger.info("Initialized OCRTool with RapidOCR engine")

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
            "results": [],  # Store OCR results
            "images": [],  # Store processed images
            "reward": 0.0,
        }

        logger.debug(f"Created OCR instance: {instance_id}")
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the OCR tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters including:
                - image_ref: Optional reference to image in shared_tool_outputs (e.g., "Extract_0")
                - use_det: Whether to use text detection (optional, default: True)
                - use_cls: Whether to use text classification (optional, default: False)
                - use_rec: Whether to use text recognition (optional, default: False)
            **kwargs: Additional parameters including:
                - image_data: List of all images (initial + tool outputs)
                - shared_tool_outputs: Dict mapping output keys to images

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing OCR results
            tool_reward_score: The step reward score of the tool
            tool_metrics: The metrics of the tool, including:
                - ocr_results: List of dicts with "box" [xmin, ymin, xmax, ymax], "text", and "confidence"
                  (box is only present when use_det=True)
        """
        # Extract parameters
        image_param = parameters.get("image") or parameters.get("image_ref")  # Support both 'image' and 'image_ref'
        use_det = parameters.get("use_det", self._default_use_det)
        use_cls = parameters.get("use_cls", self._default_use_cls)
        use_rec = parameters.get("use_rec", self._default_use_rec)

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
            logger.error(f"[OCRTool] {error_msg} image_param={image_param}, shared_keys={list(shared_tool_outputs.keys())}, image_data_len={len(image_data)}")
            return ToolResponse(text=error_msg), -0.05, {"success": False}

        try:
            # Convert PIL Image to numpy array for RapidOCR
            if isinstance(image, Image.Image):
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img_array = np.array(image)
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                error_msg = "Error: 'image' must be a PIL Image or numpy array."
                logger.error(f"[OCRTool] {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            # Perform OCR
            result, elapse = self._engine(
                img_array,
                use_det=use_det,
                use_cls=use_cls,
                use_rec=use_rec
            )

            # Parse results
            # RapidOCR returns: [[box, text, confidence], ...] or [[text, confidence], ...] when use_det=False
            ocr_results = []

            if result is not None:
                for item in result:
                    if use_det:
                        # Format: [box, text, confidence]
                        if len(item) >= 3:
                            box_points = item[0]  # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            text = str(item[1])
                            confidence = float(item[2])

                            # Convert box points to [xmin, ymin, xmax, ymax] format
                            x_coords = [point[0] for point in box_points]
                            y_coords = [point[1] for point in box_points]
                            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                            ocr_results.append({"box": box, "text": text, "confidence": confidence})
                    else:
                        # Format: [text, confidence] when use_det=False
                        if len(item) >= 2:
                            text = str(item[0])
                            confidence = float(item[1])
                            ocr_results.append({"text": text, "confidence": confidence})

            # Store results in instance dictionary
            self._instance_dict[instance_id]["results"].append(ocr_results)
            self._instance_dict[instance_id]["images"].append(image)

            # Prepare response text
            if ocr_results:
                text_lines = []
                for result in ocr_results:
                    text = result["text"]
                    conf = result["confidence"]
                    if "box" in result:
                        # Include bounding box info in response
                        box = result["box"]
                        text_lines.append(f"'{text}' (confidence: {conf:.2f}, box: {box})")
                    else:
                        text_lines.append(f"'{text}' (confidence: {conf:.2f})")
                response_text = f"OCR Results ({len(ocr_results)} text regions found):\n" + "\n".join(text_lines)
            else:
                response_text = "OCR completed but no text was detected."

            # Metrics
            metrics = {
                "success": True,
                "num_texts": len(ocr_results),
                "elapse": elapse,
                "use_det": use_det,
                "use_cls": use_cls,
                "use_rec": use_rec,
                "ocr_results": ocr_results,
            }

            logger.info(f"OCR successful for instance {instance_id}: {len(ocr_results)} text regions found")
            return ToolResponse(text=response_text), 0.0, metrics

        except Exception as e:
            error_msg = f"OCR failed: {e}"
            logger.error(f"[OCRTool] {error_msg}")
            return ToolResponse(text=error_msg), -0.05, {"success": False, "error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool (number of successfully processed images).
        """
        if instance_id not in self._instance_dict:
            return 0.0

        num_images = len(self._instance_dict[instance_id]["results"])
        return float(num_images)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            num_results = len(self._instance_dict[instance_id]["results"])
            logger.debug(f"Releasing instance {instance_id} with {num_results} OCR results")
            del self._instance_dict[instance_id]
