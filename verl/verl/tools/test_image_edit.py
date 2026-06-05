#!/usr/bin/env python3
"""
Batch image editing using ImageEditTool with LongCat-Image-Edit model
Reads prompts and images from JSON file and saves edited images
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from PIL import Image
from verl.tools.image_edit_tool import ImageEditTool
from verl.tools.schemas import OpenAIFunctionToolSchema

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Create tool configuration for LongCat Image Edit
# GPU Configuration: 8 A800 80GB GPUs available
# Using 12 workers with 0.5 GPU each - 2 workers per GPU for higher utilization
config = {
    "model_name": "/data/zhuhairui/LongCat-Image-Edit",
    "num_workers": 12,          # 12 workers, 2 per GPU
    "num_gpus_per_worker": 0.5, # Each worker gets half GPU, allowing 2 workers per GPU
    "rate_limit": 200,          # Increase to 200 requests per second
    "enable_global_rate_limit": True
}

# Define tool schema
tool_schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "edit_image",
        "description": "Edit image based on text instruction using LongCat-Image-Edit model",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text instruction for how to edit the image"
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid in the edited image (optional)"
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of denoising steps (default: 50)"
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Guidance scale (default: 4.5)"
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



def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  - Loaded {len(data)} items")
    return data


def save_json_data(json_path: str, data: List[Dict[str, Any]]):
    """Save updated JSON data to file."""
    output_path = json_path.replace('.json', '_edited.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def save_json_data_incremental(json_path: str, data: List[Dict[str, Any]]):
    """Save JSON data incrementally without printing (for frequent saves)."""
    output_path = json_path.replace('.json', '_edited.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


async def main():
    # Configuration
    INPUT_JSON = "/nfsdata4/zhuhairui/EDIT/processed_data_20k.json"
    OUTPUT_DIR = "/nfsdata4/zhuhairui/EDIT/edited_images"
    
    print("=" * 60)
    print("Batch Image Editing - LongCat-Image-Edit")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[Setup] Output directory: {OUTPUT_DIR}")
    
    # Load JSON data
    print("\n[1/6] Loading JSON data...")
    data = load_json_data(INPUT_JSON)
    
    # Create tool instance
    print("\n[2/6] Creating ImageEditTool instance...")
    tool = ImageEditTool(config, tool_schema)
    print(f"  - Model: {config['model_name']}")
    print(f"  - Workers: {config['num_workers']}")
    print(f"  - GPU per worker: {config['num_gpus_per_worker']}")
    print(f"  - Total GPU(s) allocated: {config['num_workers'] * config['num_gpus_per_worker']}")
    print(f"  - Rate limit: {config['rate_limit']} req/sec")
    
    # Create instance
    print("\n[3/6] Creating tool instance...")
    instance_id, creation_response = await tool.create()
    print(f"  - Instance ID: {instance_id}")
    
    # Process each item
    print("\n[4/6] Processing images...")
    total_items = len(data)
    success_count = 0
    error_count = 0
    
    # Create a lock for thread-safe JSON writes
    import threading
    json_lock = threading.Lock()
    
    # Create a queue for concurrent processing
    from asyncio import Semaphore
    semaphore = Semaphore(12)  # Limit to 12 concurrent tasks (2 per GPU)
    
    async def process_item(idx: int, item: Dict[str, Any]) -> bool:
        """Process a single item with semaphore limiting."""
        async with semaphore:
            try:
                prompt = item.get('prompt', '')
                image_paths = item.get('image', [])
                
                if not prompt or not image_paths:
                    print(f"  [{idx+1}/{total_items}] ⊘ Skipping - missing prompt or image")
                    return False
                
                # Load original image
                original_image_path = image_paths[0]
                if not os.path.exists(original_image_path):
                    print(f"  [{idx+1}/{total_items}] ✗ Error: Image not found")
                    return False
                
                original_image = Image.open(original_image_path).convert('RGB')
                
                print(f"  [{idx+1}/{total_items}] Processing: {Path(original_image_path).name}")
                
                # Edit image
                response, reward, metrics = await tool.execute(
                    instance_id,
                    {
                        "image": original_image,
                        "prompt": prompt,
                        "num_inference_steps": 50,
                        "guidance_scale": 4.5,
                        "negative_prompt": "",
                        "seed": 42
                    }
                )
                
                # Save edited image
                if response.image and len(response.image) > 0:
                    # Generate output filename
                    original_name = Path(original_image_path).stem
                    output_filename = f"{original_name}_edited_{idx:05d}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    # Save image
                    response.image[0].save(output_path)
                    
                    # Add edited image path to the image list and save JSON immediately
                    with json_lock:
                        item['image'].append(output_path)
                        save_json_data_incremental(INPUT_JSON, data)
                    
                    print(f"  [{idx+1}/{total_items}] ✓ Success (reward: {reward:.4f}) - JSON saved")
                    return True
                else:
                    print(f"  [{idx+1}/{total_items}] ✗ No image generated")
                    return False
                    
            except Exception as e:
                print(f"  [{idx+1}/{total_items}] ✗ Error: {str(e)[:50]}")
                return False
    
    # Create tasks for all items
    print(f"  Starting parallel processing with 12 workers (2 per GPU)...")
    tasks = [process_item(idx, item) for idx, item in enumerate(data)]
    
    # Run all tasks and count results
    results = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r is True)
    error_count = sum(1 for r in results if r is False)
    
    # Save updated JSON
    print("\n[5/6] Final JSON save...")
    output_json_path = save_json_data(INPUT_JSON, data)
    print(f"  - Saved to: {output_json_path}")
    
    # Cleanup
    print("\n[6/6] Releasing instance...")
    await tool.release(instance_id)
    print("  ✓ Instance released")
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total items:           {total_items}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors:                {error_count}")
    print(f"Success rate:          {success_count/total_items*100:.1f}%")
    print(f"\nGPU Configuration:")
    print(f"  - Workers used:      {config['num_workers']}")
    print(f"  - Total GPUs:        {config['num_workers']} × {config['num_gpus_per_worker']}")
    print(f"  - Rate limit:        {config['rate_limit']} req/sec")
    print(f"\nOutput:")
    print(f"  - Updated JSON: {output_json_path}")
    print(f"  - Edited images: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
