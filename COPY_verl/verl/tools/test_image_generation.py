#!/usr/bin/env python3
"""
Standalone test for ImageGenerationTool with ZImage Turbo model
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from verl.tools.image_generation_tool import ImageGenerationTool
from verl.tools.schemas import OpenAIFunctionToolSchema

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Create tool configuration for ZImage Turbo
config = {
    "model_name": "/data/zhuhairui/Z-Image-Turbo",
    "num_workers": 1,
    "num_gpus_per_worker": 1.0,
    "rate_limit": 5,
    "enable_global_rate_limit": True
}

# Define tool schema
tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate image from text prompt using ZImage Turbo model",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate"
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of denoising steps (default: 9 for Turbo)"
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Guidance scale (default: 0.0 for Turbo)"
                },
                "height": {
                    "type": "integer",
                    "description": "Image height (default: 1024)"
                },
                "width": {
                    "type": "integer",
                    "description": "Image width (default: 1024)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": ["prompt"]
        }
    }
})

async def main():
    print("=" * 60)
    print("ImageGenerationTool Test - ZImage Turbo")
    print("=" * 60)
    
    # Create tool instance
    print("\n[1/4] Creating ImageGenerationTool instance...")
    tool = ImageGenerationTool(config, tool_schema)
    print(f"  - Model: {config['model_name']}")
    print(f"  - Workers: {config['num_workers']}")
    print(f"  - GPU per worker: {config['num_gpus_per_worker']}")
    
    # Create instance
    print("\n[2/4] Creating tool instance...")
    instance_id, creation_response = await tool.create()
    print(f"  - Instance ID: {instance_id}")
    
    # Generate image
    print("\n[3/4] Generating image...")
    prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."
    
    print(f"  - Prompt: {prompt[:80]}...")
    
    response, reward, metrics = await tool.execute(
        instance_id,
        {
            "prompt": prompt,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "height": 1024,
            "width": 1024,
            "seed": 42
        }
    )
    
    print(f"\n  Results:")
    print(f"  - Response text: {response.text[:100]}..." if response.text else "  - Response text: None")
    print(f"  - Reward: {reward}")
    print(f"  - Metrics: {metrics}")
    
    # Save image
    if response.image:
        output_path = "test_generated_image.png"
        response.image[0].save(output_path)
        print(f"\n  ✓ Image saved to: {output_path}")
    else:
        print("\n  ✗ No image generated")
    
    # Cleanup
    print("\n[4/4] Releasing instance...")
    await tool.release(instance_id)
    print("  ✓ Instance released")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
