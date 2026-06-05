"""Test script for OCRTool"""
import asyncio
from PIL import Image

# Add parent path for imports
import sys
sys.path.insert(0, '/data/zhuhairui/verl')

from verl.tools.ocr_tool import OCRTool
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema


async def test_ocr_tool():
    # Create tool configuration
    config = {
        "type": "native",
        "use_det": True,
        "use_cls": True,
        "use_rec": True,
    }
    
    # Create proper schema using Pydantic models
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="ocr",
            description="Performs OCR on an image and returns recognized text with confidence scores",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={},
                required=[]
            )
        )
    )
    
    # Initialize OCR tool
    print("Initializing OCR tool...")
    ocr_tool = OCRTool(config, tool_schema)
    
    # Create instance
    instance_id, _ = await ocr_tool.create()
    print(f"Created instance: {instance_id}")
    
    # Load test image
    img_path = '/data/zhuhairui/verl/verl/tools/output_0.png'
    print(f"Loading image: {img_path}")
    image = Image.open(img_path)
    
    # Test 1: Full OCR (detection + classification + recognition)
    print("\n=== Test 1: Full OCR ===")
    response, reward, metrics = await ocr_tool.execute(
        instance_id,
        {"image": image, "use_det": True, "use_cls": True, "use_rec": True}
    )
    print(f"Response: {response.text}")
    print(f"OCR Results: {metrics.get('ocr_results', [])}")
    print(f"Elapse: {metrics.get('elapse')}")
    
    # Test 2: Recognition only (no detection)
    print("\n=== Test 2: Recognition Only ===")
    response2, reward2, metrics2 = await ocr_tool.execute(
        instance_id,
        {"image": image, "use_det": False, "use_cls": False, "use_rec": True}
    )
    print(f"Response: {response2.text}")
    print(f"OCR Results: {metrics2.get('ocr_results', [])}")
    
    # Calculate reward
    total_reward = await ocr_tool.calc_reward(instance_id)
    print(f"\nTotal reward: {total_reward}")
    
    # Release instance
    await ocr_tool.release(instance_id)
    print("Instance released")


if __name__ == "__main__":
    asyncio.run(test_ocr_tool())
