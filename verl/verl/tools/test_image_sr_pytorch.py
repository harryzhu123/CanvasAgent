#!/usr/bin/env python3
"""
Test script for ImageSRToolPyTorch (Optimized PyTorch version)
Compares performance with the original ncnn-vulkan version
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import ray
from PIL import Image
from verl.tools.image_sr_tool_pytorch import ImageSRToolPyTorch
from verl.tools.schemas import OpenAIFunctionToolSchema

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Tool configuration for PyTorch version (optimized)
config = {
    "model_name": "RealESRGAN_x4plus",
    "scale": 4,
    "tile_size": 0,  # 0 = no tiling (fastest)
    "num_workers": 4,  # More workers possible due to lower memory usage
    "num_gpus_per_worker": 0.25,  # Each worker uses 1/4 GPU
    "rate_limit": 40,
    "enable_global_rate_limit": True
}

# Define tool schema
tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "super_resolve",
        "description": "Upscales an image using Real-ESRGAN PyTorch (4x, optimized)",
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


async def test_pytorch_sr():
    """Test PyTorch-based super resolution (optimized version)."""
    print("=" * 70)
    print("ImageSRToolPyTorch - Optimized Performance Test")
    print("=" * 70)

    # Test image path
    test_image_path = "/data/zhuhairui/verl/verl/tools/dog-3.jpg"
    output_dir = "/data/zhuhairui/verl/verl/tools"

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found: {test_image_path}")
        return

    print(f"\n[1/6] Loading test image...")
    print(f"  - Input: {test_image_path}")
    original_image = Image.open(test_image_path).convert('RGB')
    print(f"  - Original size: {original_image.size[0]}x{original_image.size[1]}")

    # Create tool instance
    print(f"\n[2/6] Creating ImageSRToolPyTorch instance...")
    print(f"  - Model: {config['model_name']}")
    print(f"  - Scale: {config['scale']}x")
    print(f"  - Workers: {config['num_workers']}")
    print(f"  - GPU per worker: {config['num_gpus_per_worker']}")
    print(f"  - Total GPU allocation: {config['num_workers'] * config['num_gpus_per_worker']}")

    tool = ImageSRToolPyTorch(config, tool_schema)

    # Create instance
    print(f"\n[3/6] Creating tool instance...")
    instance_id, creation_response = await tool.create(image=original_image)
    print(f"  - Instance ID: {instance_id}")

    # First execution (includes model loading time)
    print(f"\n[4/6] First execution (includes model loading)...")
    start_time = time.time()
    try:
        response, reward, metrics = await tool.execute(
            instance_id,
            {},
            image_data=[original_image]
        )
        first_exec_time = time.time() - start_time

        print(f"  - Status: {metrics.get('success', False)}")
        print(f"  - Backend: {metrics.get('backend', 'unknown')}")
        print(f"  - Time: {first_exec_time:.2f}s (includes model loading)")
        print(f"  - Response: {response.text}")

        # Save first result
        if response.image and len(response.image) > 0:
            output_path = os.path.join(output_dir, "dog-3_pytorch_first.png")
            response.image[0].save(output_path)
            print(f"  - Saved to: {output_path}")

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Second execution (model already loaded)
    print(f"\n[5/6] Second execution (model cached in GPU)...")
    start_time = time.time()
    try:
        response, reward, metrics = await tool.execute(
            instance_id,
            {},
            image_data=[original_image]
        )
        second_exec_time = time.time() - start_time

        print(f"  - Status: {metrics.get('success', False)}")
        print(f"  - Time: {second_exec_time:.2f}s (model already loaded)")
        print(f"  - Speedup vs first: {first_exec_time/second_exec_time:.1f}x faster")

        # Save second result
        if response.image and len(response.image) > 0:
            output_path = os.path.join(output_dir, "dog-3_pytorch_second.png")
            response.image[0].save(output_path)

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()

    # Benchmark: Multiple runs
    print(f"\n[6/6] Benchmark: 5 consecutive runs...")
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response, reward, metrics = await tool.execute(
                instance_id,
                {},
                image_data=[original_image]
            )
            exec_time = time.time() - start_time
            times.append(exec_time)
            print(f"  Run {i+1}/5: {exec_time:.3f}s")
        except Exception as e:
            print(f"  Run {i+1}/5: Error - {e}")

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\n  Benchmark Results:")
        print(f"    - Average: {avg_time:.3f}s")
        print(f"    - Min: {min_time:.3f}s")
        print(f"    - Max: {max_time:.3f}s")
        print(f"    - Throughput: ~{1/avg_time:.1f} images/sec per worker")
        print(f"    - Total throughput: ~{config['num_workers']/avg_time:.1f} images/sec")

    # Cleanup
    print(f"\n[7/6] Releasing instance...")
    await tool.release(instance_id)
    print("  ✓ Instance released")

    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"PyTorch (this test):")
    print(f"  - First run (with loading): {first_exec_time:.2f}s")
    print(f"  - Subsequent runs: ~{avg_time:.3f}s")
    print(f"  - Model: Loaded once, kept in GPU memory")
    print(f"\nOriginal ncnn-vulkan version:")
    print(f"  - Every run: ~3-5s")
    print(f"  - Model: Reloaded every time")
    print(f"\nExpected speedup: ~{3/avg_time:.0f}x - {5/avg_time:.0f}x faster")
    print("=" * 70)


async def test_batch_processing():
    """Test batch processing with multiple images."""
    print("\n" + "=" * 70)
    print("Batch Processing Test - Multiple Images")
    print("=" * 70)

    test_images = [
        "/data/zhuhairui/verl/verl/tools/dog-3.jpg",
        "/data/zhuhairui/verl/verl/tools/realesrgan-ncnn-vulkan/input.jpg",
        "/data/zhuhairui/verl/verl/tools/realesrgan-ncnn-vulkan/input2.jpg",
    ]

    # Filter existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    print(f"\nFound {len(existing_images)} test images")

    if len(existing_images) == 0:
        print("No test images found, skipping batch test")
        return

    # Create tool
    print(f"\nCreating ImageSRToolPyTorch...")
    tool = ImageSRToolPyTorch(config, tool_schema)

    # Process all images
    print(f"\nProcessing {len(existing_images)} images in parallel...")
    start_time = time.time()

    tasks = []
    for idx, img_path in enumerate(existing_images):
        # Load image
        image = Image.open(img_path).convert('RGB')

        # Create instance
        instance_id = f"batch_{idx}"
        await tool.create(instance_id=instance_id, image=image)

        # Create task
        task = tool.execute(instance_id, {}, image_data=[image])
        tasks.append((instance_id, Path(img_path).name, task))

    # Execute all tasks
    results = []
    for instance_id, img_name, task in tasks:
        try:
            response, reward, metrics = await task
            results.append((img_name, metrics.get('success', False)))
            print(f"  ✓ {img_name}: {metrics['original_size']} -> {metrics['output_size']}")
        except Exception as e:
            results.append((img_name, False))
            print(f"  ✗ {img_name}: Error - {e}")

        # Cleanup
        await tool.release(instance_id)

    total_time = time.time() - start_time
    successful = sum(1 for _, success in results if success)

    print(f"\nBatch Processing Summary:")
    print(f"  - Total images: {len(existing_images)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {len(existing_images) - successful}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average per image: {total_time/len(existing_images):.2f}s")
    print("=" * 70)


async def main():
    """Run all tests."""
    try:
        # Test 1: Basic performance test
        await test_pytorch_sr()

        # Test 2: Batch processing
        # await test_batch_processing()

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            print("\nShutting down Ray...")
            ray.shutdown()
            print("✓ Ray shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
