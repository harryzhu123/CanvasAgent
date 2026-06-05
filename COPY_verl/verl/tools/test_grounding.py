"""Test script for ImageGroundingTool"""
import asyncio
from PIL import Image

# Add parent path for imports
import sys
sys.path.insert(0, '/data/zhuhairui/verl')

from verl.tools.image_grounding_tool import ImageGroundingTool
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema


async def test_grounding_tool():
    # Create tool configuration
    config = {
        "type": "native",
        "model_dir": "/data/zhuhairui/GroundingDINO",
        "num_workers": 1,
        "num_gpus_per_worker": 0.25,
        "rate_limit": 10,
        "enable_global_rate_limit": False,  # Disable for testing
    }
    
    # Create proper schema using Pydantic models
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="grounding",
            description="Locates objects in an image based on text description and returns bounding boxes",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "reference_text": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Text describing objects to locate"
                    )
                },
                required=["reference_text"]
            )
        )
    )
    
    # Initialize Grounding tool
    print("Initializing ImageGroundingTool...")
    grounding_tool = ImageGroundingTool(config, tool_schema)
    
    # Create instance
    instance_id, _ = await grounding_tool.create()
    print(f"Created instance: {instance_id}")
    
    # Load test image
    img_path = '/data/zhuhairui/LongCat-Image-Edit/assets/test.png'
    print(f"Loading image: {img_path}")
    image = Image.open(img_path)
    
    # Test grounding
    print("\n=== Test: Object Grounding ===")
    response, reward, metrics = await grounding_tool.execute(
        instance_id,
        {
            "image": image,
            "reference_text": "cat . dog . animal . person .",
            "box_threshold": 0.35,
            "text_threshold": 0.25
        }
    )
    print(f"Response:\n{response.text}")
    print(f"Metrics: {metrics}")
    
    # Calculate reward
    total_reward = await grounding_tool.calc_reward(instance_id)
    print(f"\nTotal reward: {total_reward}")
    
    # Release instance
    await grounding_tool.release(instance_id)
    print("Instance released")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    asyncio.run(test_grounding_tool())
