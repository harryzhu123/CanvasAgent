#!/usr/bin/env python3
"""
Standalone test for ImageRotateTool and ImageFlipTool
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from verl.tools.image_rotate_tool import ImageRotateTool
from verl.tools.image_flip_tool import ImageFlipTool
from verl.tools.schemas import OpenAIFunctionToolSchema


def create_test_image():
    """Create a simple test image."""
    # Check if test image exists
    test_image_path = "/data/zhuhairui/LongCat-Image-Edit/assets/test.png"
    if os.path.exists(test_image_path):
        print(f"  Using existing test image: {test_image_path}")
        return Image.open(test_image_path).convert('RGB')
    
    # Create a synthetic test image with an arrow to show orientation
    print("  Creating synthetic test image...")
    import numpy as np
    
    width, height = 512, 512
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue gradient background
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [100, 150, 200]
    
    # Draw a red arrow pointing right to show orientation
    # Arrow body
    for y in range(height//2 - 20, height//2 + 20):
        for x in range(100, 350):
            img_array[y, x] = [255, 0, 0]
    
    # Arrow head
    for i in range(60):
        for j in range(-i, i+1):
            y = height//2 + j
            x = 350 + i
            if 0 <= y < height and 0 <= x < width:
                img_array[y, x] = [255, 0, 0]
    
    return Image.fromarray(img_array)


# Define tool schemas
rotate_tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "rotate_image",
        "description": "Rotates an image by the specified angle",
        "parameters": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Rotation angle in degrees (positive = counter-clockwise)"
                },
                "expand": {
                    "type": "boolean",
                    "description": "Whether to expand canvas to fit rotated image (default: true)"
                }
            },
            "required": ["angle"]
        }
    }
})

flip_tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "flip_image",
        "description": "Flips an image horizontally or vertically",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                    "description": "Direction of flip"
                }
            },
            "required": ["direction"]
        }
    }
})


async def test_rotate_tool():
    """Test the ImageRotateTool."""
    print("\n" + "=" * 60)
    print("Testing ImageRotateTool")
    print("=" * 60)
    
    # Create tool instance
    print("\n[1/4] Creating ImageRotateTool instance...")
    config = {}
    tool = ImageRotateTool(config, rotate_tool_schema)
    
    # Create instance
    print("[2/4] Creating tool instance...")
    instance_id, _ = await tool.create()
    print(f"  - Instance ID: {instance_id}")
    
    # Load test image
    print("[3/4] Loading test image...")
    test_image = create_test_image()
    print(f"  - Image size: {test_image.size}")
    test_image.save("test_rotate_original.png")
    print("  - Saved: test_rotate_original.png")
    
    # Test rotations
    print("[4/4] Testing rotations...")
    
    test_cases = [
        {"angle": 90, "name": "90°"},
        {"angle": 180, "name": "180°"},
        {"angle": 45, "name": "45°"},
    ]
    
    for tc in test_cases:
        response, reward, metrics = await tool.execute(
            instance_id,
            {
                "image": test_image,
                "angle": tc["angle"],
                "expand": True
            }
        )
        
        if response.image:
            output_path = f"test_rotate_{tc['angle']}.png"
            response.image[0].save(output_path)
            print(f"  ✓ Rotated {tc['name']}: {output_path} (size: {metrics.get('rotated_size')})")
        else:
            print(f"  ✗ Failed to rotate {tc['name']}: {response.text}")
    
    # Cleanup
    await tool.release(instance_id)
    print("\nImageRotateTool test completed!")


async def test_flip_tool():
    """Test the ImageFlipTool."""
    print("\n" + "=" * 60)
    print("Testing ImageFlipTool")
    print("=" * 60)
    
    # Create tool instance
    print("\n[1/4] Creating ImageFlipTool instance...")
    config = {}
    tool = ImageFlipTool(config, flip_tool_schema)
    
    # Create instance
    print("[2/4] Creating tool instance...")
    instance_id, _ = await tool.create()
    print(f"  - Instance ID: {instance_id}")
    
    # Load test image
    print("[3/4] Loading test image...")
    test_image = create_test_image()
    print(f"  - Image size: {test_image.size}")
    test_image.save("test_flip_original.png")
    print("  - Saved: test_flip_original.png")
    
    # Test flips
    print("[4/4] Testing flips...")
    
    test_cases = [
        {"direction": "horizontal", "name": "Horizontal"},
        {"direction": "vertical", "name": "Vertical"},
    ]
    
    for tc in test_cases:
        response, reward, metrics = await tool.execute(
            instance_id,
            {
                "image": test_image,
                "direction": tc["direction"]
            }
        )
        
        if response.image:
            output_path = f"test_flip_{tc['direction']}.png"
            response.image[0].save(output_path)
            print(f"  ✓ Flipped {tc['name']}: {output_path}")
        else:
            print(f"  ✗ Failed to flip {tc['name']}: {response.text}")
    
    # Cleanup
    await tool.release(instance_id)
    print("\nImageFlipTool test completed!")


async def main():
    print("=" * 60)
    print("Image Rotate & Flip Tools Test")
    print("=" * 60)
    
    await test_rotate_tool()
    await test_flip_tool()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in os.listdir('.'):
        if f.startswith('test_rotate_') or f.startswith('test_flip_'):
            print(f"  - {f}")


if __name__ == "__main__":
    asyncio.run(main())
