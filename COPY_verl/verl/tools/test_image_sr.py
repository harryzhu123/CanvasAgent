#!/usr/bin/env python3
"""
Simple test script for ImageSRTool (Image Super Resolution)
Tests basic upscaling functionality using realesrgan-ncnn-vulkan
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
# Script is at: verl/tools/test_image_sr.py, need to go up 2 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import ray
from PIL import Image
from verl.tools.image_sr_tool import ImageSRTool
from verl.tools.schemas import OpenAIFunctionToolSchema

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Tool configuration for ImageSRTool
config = {
    "executable_path": "/data/zhuhairui/verl/verl/tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan",
    "model_name": "realesrgan-x4plus",  # Options: realesrgan-x4plus, realesrgan-x4plus-anime, realesr-animevideov3
    "scale": 4,
    "num_workers": 2,
    "gpu_id": 0,  # GPU device ID (-1 for CPU)
    "rate_limit": 20,
    "enable_global_rate_limit": True
}

# Define tool schema
tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "super_resolve",
        "description": "Upscales an image using Real-ESRGAN (4x)",
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "Reference to image in shared_tool_outputs"
                }
            },
            "required": []
        }
    }
})


async def test_basic_sr():
    """Test basic super resolution functionality."""
    print("=" * 60)
    print("ImageSRTool - Basic Test")
    print("=" * 60)

    # Test image path (using existing test image)
    test_image_path = "/data/zhuhairui/verl/verl/tools/dog-3.jpg"
    output_dir = "/data/zhuhairui/verl/verl/tools"

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found: {test_image_path}")
        return

    print(f"\n[1/5] Loading test image...")
    print(f"  - Input: {test_image_path}")
    original_image = Image.open(test_image_path).convert('RGB')
    print(f"  - Original size: {original_image.size[0]}x{original_image.size[1]}")

    # Create tool instance
    print(f"\n[2/5] Creating ImageSRTool instance...")
    tool = ImageSRTool(config, tool_schema)
    print(f"  - Model: {config['model_name']}")
    print(f"  - Scale: {config['scale']}x")
    print(f"  - Workers: {config['num_workers']}")
    print(f"  - GPU ID: {config['gpu_id']}")

    # Create instance
    print(f"\n[3/5] Creating tool instance...")
    instance_id, creation_response = await tool.create(image=original_image)
    print(f"  - Instance ID: {instance_id}")

    # Execute super resolution
    print(f"\n[4/5] Executing super resolution...")
    try:
        response, reward, metrics = await tool.execute(
            instance_id,
            {},  # Empty parameters - will use image from create()
            image_data=[original_image]  # Also provide via image_data
        )

        print(f"  - Status: {metrics.get('success', False)}")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Response: {response.text}")

        # Save result
        if response.image and len(response.image) > 0:
            output_path = os.path.join(output_dir, "dog-3_sr_output.png")
            response.image[0].save(output_path)
            print(f"  - Saved SR image to: {output_path}")

            # Display metrics
            print(f"\n  Metrics:")
            print(f"    - Original: {metrics['original_size'][0]}x{metrics['original_size'][1]}")
            print(f"    - Upscaled: {metrics['output_size'][0]}x{metrics['output_size'][1]}")
            print(f"    - Scale: {metrics['scale']}x")
            print(f"    - Worker ID: {metrics['worker_id']}")
        else:
            print("  - Warning: No image generated")

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    print(f"\n[5/5] Releasing instance...")
    await tool.release(instance_id)
    print("  ✓ Instance released")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


async def test_multiple_images():
    """Test processing multiple images."""
    print("\n" + "=" * 60)
    print("ImageSRTool - Multiple Images Test")
    print("=" * 60)

    # Test images
    test_images = [
        "/data/zhuhairui/verl/verl/tools/dog-3.jpg",
        "/data/zhuhairui/verl/verl/tools/realesrgan-ncnn-vulkan/input.jpg",
    ]

    output_dir = "/data/zhuhairui/verl/verl/tools"

    # Filter existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    print(f"\n[1/4] Found {len(existing_images)} test images")

    # Create tool
    print(f"\n[2/4] Creating ImageSRTool...")
    tool = ImageSRTool(config, tool_schema)

    # Process each image
    print(f"\n[3/4] Processing {len(existing_images)} images...")
    results = []

    for idx, img_path in enumerate(existing_images):
        print(f"\n  Image {idx+1}/{len(existing_images)}: {Path(img_path).name}")

        # Load image
        image = Image.open(img_path).convert('RGB')
        print(f"    - Size: {image.size[0]}x{image.size[1]}")

        # Create instance
        instance_id, _ = await tool.create(image=image)

        # Execute
        try:
            response, reward, metrics = await tool.execute(
                instance_id,
                {},
                image_data=[image]
            )

            if response.image and len(response.image) > 0:
                # Save
                output_filename = f"{Path(img_path).stem}_sr_{idx}.png"
                output_path = os.path.join(output_dir, output_filename)
                response.image[0].save(output_path)

                print(f"    ✓ Success: {metrics['original_size']} -> {metrics['output_size']}")
                print(f"    - Saved: {output_filename}")
                results.append(True)
            else:
                print(f"    ✗ Failed: No image generated")
                results.append(False)

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append(False)

        # Cleanup
        await tool.release(instance_id)

    # Summary
    print(f"\n[4/4] Summary")
    print(f"  - Total: {len(existing_images)}")
    print(f"  - Success: {sum(results)}")
    print(f"  - Failed: {len(results) - sum(results)}")

    print("\n" + "=" * 60)


async def main():
    """Run all tests."""
    try:
        # Test 1: Basic SR
        await test_basic_sr()

        # Test 2: Multiple images
        # await test_multiple_images()

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
