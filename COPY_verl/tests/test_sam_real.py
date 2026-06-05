"""Real end-to-end test for ImageSAMTool with Ray + GPU + SAM model."""

import asyncio
import numpy as np
from PIL import Image

MODEL_PATH = "/data/zhuhairui/sam_vit_h.pth"


def make_tool_schema():
    from verl.tools.schemas import OpenAIFunctionToolSchema
    return OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "segment",
            "description": "Segment objects using SAM",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "array",
                        "description": "Bounding box [x_min, y_min, x_max, y_max] (0-1000 normalized)",
                    }
                },
                "required": ["bbox"],
            },
        },
    )


def make_test_image(width=640, height=480):
    """Create a test image with a colored rectangle for segmentation."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [200, 200, 200]  # gray background
    img[100:350, 150:500, :] = [255, 0, 0]  # red rectangle
    return Image.fromarray(img, "RGB")


async def run_test():
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from verl.tools.image_sam_tool import ImageSAMTool

    config = {
        "model_path": MODEL_PATH,
        "model_type": "vit_h",
        "num_workers": 1,
        "num_gpus_per_worker": 0.5,
        "rate_limit": 5,
    }

    print("=" * 60)
    print("Initializing ImageSAMTool...")
    tool = ImageSAMTool(config, make_tool_schema())

    # --- Test 1: basic segmentation ---
    print("\n[Test 1] Basic segmentation with list bbox")
    instance_id, _ = await tool.create()
    img = make_test_image()
    print(f"  Image size: {img.size}")

    # bbox in 0-1000 normalized coords targeting the red rectangle area
    # red rect is at pixel (150,100)-(500,350) in a 640x480 image
    # normalized: x_min=150/640*1000=234, y_min=100/480*1000=208, x_max=500/640*1000=781, y_max=350/480*1000=729
    bbox = [234, 208, 781, 729]
    print(f"  Bounding box (normalized 0-1000): {bbox}")

    resp, reward, metrics = await tool.execute(
        instance_id, parameters={"bbox": bbox}, image_data=[img]
    )
    print(f"  Success: {metrics.get('success')}")
    print(f"  Score: {metrics.get('score', 'N/A')}")
    print(f"  Mask shape: {metrics.get('mask_shape')}")
    print(f"  Mask area: {metrics.get('mask_area')} ({metrics.get('mask_ratio', 0):.2%})")
    print(f"  Response text: {resp.text[:120]}...")
    assert metrics["success"] is True, "Test 1 FAILED"
    assert resp.image is not None and len(resp.image) == 1, "No mask image returned"
    assert resp.image[0].mode == "L", f"Expected mode 'L', got '{resp.image[0].mode}'"
    print("  -> PASSED")
    await tool.release(instance_id)

    # --- Test 2: string bbox format ---
    print("\n[Test 2] String bbox format")
    instance_id, _ = await tool.create()
    resp, reward, metrics = await tool.execute(
        instance_id, parameters={"bbox": "[234,208,781,729]"}, image_data=[img]
    )
    print(f"  Success: {metrics.get('success')}")
    print(f"  Score: {metrics.get('score', 'N/A')}")
    assert metrics["success"] is True, "Test 2 FAILED"
    print("  -> PASSED")
    await tool.release(instance_id)

    # --- Test 3: image via shared_tool_outputs ---
    print("\n[Test 3] Image from shared_tool_outputs")
    instance_id, _ = await tool.create()
    resp, reward, metrics = await tool.execute(
        instance_id,
        parameters={"bbox": [234, 208, 781, 729], "image_ref": "Crop_0"},
        shared_tool_outputs={"Crop_0": img},
    )
    print(f"  Success: {metrics.get('success')}")
    assert metrics["success"] is True, "Test 3 FAILED"
    print("  -> PASSED")
    await tool.release(instance_id)

    # --- Test 4: calc_reward ---
    print("\n[Test 4] calc_reward after segmentation")
    instance_id, _ = await tool.create()
    await tool.execute(instance_id, parameters={"bbox": [234, 208, 781, 729]}, image_data=[img])
    reward = await tool.calc_reward(instance_id)
    print(f"  Reward after 1 segmentation: {reward}")
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    print("  -> PASSED")
    await tool.release(instance_id)

    # --- Test 5: error cases ---
    print("\n[Test 5] Error: missing bbox")
    instance_id, _ = await tool.create()
    resp, reward, metrics = await tool.execute(instance_id, parameters={}, image_data=[img])
    print(f"  Success: {metrics.get('success')} (expected False)")
    assert metrics["success"] is False, "Test 5 FAILED"
    print("  -> PASSED")
    await tool.release(instance_id)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(run_test())
