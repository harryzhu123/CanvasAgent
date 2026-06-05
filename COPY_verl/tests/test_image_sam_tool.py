"""Simple tests for ImageSAMTool.

Tests cover:
1. Unit tests (no Ray/GPU needed): parameter parsing, instance lifecycle
2. Integration test (requires Ray + GPU + SAM model): end-to-end segmentation

Run: python -m pytest tests/test_image_sam_tool.py -v --tb=short
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_schema():
    """Create a minimal OpenAI tool schema for SAM."""
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
                        "description": "Bounding box [x_min, y_min, x_max, y_max]",
                    }
                },
                "required": ["bbox"],
            },
        },
    )


def _make_dummy_image(width=100, height=100):
    """Create a simple dummy RGB image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _fake_sam_result(h=100, w=100):
    """Create a fake SAM segmentation result."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    return {
        "mask": mask.tolist(),
        "score": 0.95,
        "mask_shape": [h, w],
        "image_size": [w, h],
        "bounding_box": [20, 20, 80, 80],
    }


def _make_tool():
    """Create an ImageSAMTool with mocked Ray pool."""
    with patch("verl.tools.image_sam_tool.init_sam_execution_pool", return_value=[MagicMock()]):
        from verl.tools.image_sam_tool import ImageSAMTool
        config = {"model_path": "/fake/path.pth", "num_workers": 1}
        return ImageSAMTool(config, _make_tool_schema())


def _make_fake_worker():
    """Create a mock worker that returns a fake SAM result."""
    w = MagicMock()
    fut = asyncio.Future()
    fut.set_result(_fake_sam_result())
    w.execute.remote = MagicMock(return_value=fut)
    return w


# ---------------------------------------------------------------------------
# Unit tests – no Ray / GPU required
# ---------------------------------------------------------------------------

class TestParameterParsing:
    """Test bbox parameter parsing and instance lifecycle without Ray."""

    def test_schema(self):
        tool = _make_tool()
        schema = tool.get_openai_tool_schema()
        assert schema.function.name == "segment"

    def test_create_and_release(self):
        tool = _make_tool()
        instance_id, resp = asyncio.get_event_loop().run_until_complete(tool.create())
        assert instance_id is not None
        assert isinstance(resp, ToolResponse)
        assert instance_id in tool._instance_dict

        asyncio.get_event_loop().run_until_complete(tool.release(instance_id))
        assert instance_id not in tool._instance_dict

    def test_missing_bbox(self):
        tool = _make_tool()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={}, image_data=[_make_dummy_image()])
        )
        assert "missing" in resp.text.lower() or "error" in resp.text.lower()
        assert reward < 0
        assert metrics["success"] is False

    def test_bbox_string_parsing(self):
        """String bbox like '[100,200,300,400]' is parsed correctly."""
        tool = _make_tool()
        tool.execution_pool = [_make_fake_worker()]

        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": "[100,200,300,400]"}, image_data=[_make_dummy_image()])
        )
        assert metrics["success"] is True
        assert "completed" in resp.text.lower()

    def test_bbox_list_parsing(self):
        """List bbox like [100, 200, 300, 400] works."""
        tool = _make_tool()
        tool.execution_pool = [_make_fake_worker()]

        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [100, 200, 300, 400]}, image_data=[_make_dummy_image()])
        )
        assert metrics["success"] is True

    def test_bbox_wrong_length(self):
        tool = _make_tool()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [1, 2, 3]}, image_data=[_make_dummy_image()])
        )
        assert metrics["success"] is False

    def test_no_image(self):
        tool = _make_tool()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [10, 20, 30, 40]})
        )
        assert metrics["success"] is False
        assert "no image" in resp.text.lower()

    def test_image_from_shared_tool_outputs(self):
        """Image resolved via shared_tool_outputs with image_ref key."""
        tool = _make_tool()
        tool.execution_pool = [_make_fake_worker()]

        img = _make_dummy_image()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(
                instance_id,
                parameters={"bbox": [100, 200, 300, 400], "image_ref": "Crop_0"},
                shared_tool_outputs={"Crop_0": img},
            )
        )
        assert metrics["success"] is True

    def test_mask_output_is_pil_image(self):
        """ToolResponse should contain a PIL mask image in mode 'L'."""
        tool = _make_tool()
        tool.execution_pool = [_make_fake_worker()]

        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        resp, _, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [100, 200, 300, 400]}, image_data=[_make_dummy_image()])
        )
        assert resp.image is not None
        assert len(resp.image) == 1
        assert isinstance(resp.image[0], Image.Image)
        assert resp.image[0].mode == "L"

    def test_calc_reward(self):
        tool = _make_tool()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())

        reward = asyncio.get_event_loop().run_until_complete(tool.calc_reward(instance_id))
        assert reward == 0.0

        tool._instance_dict[instance_id]["results"].append(_fake_sam_result())
        reward = asyncio.get_event_loop().run_until_complete(tool.calc_reward(instance_id))
        assert reward == 1.0

    def test_get_mask_from_instance(self):
        tool = _make_tool()
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        assert tool.get_mask_from_instance(instance_id) is None

        tool._instance_dict[instance_id]["results"].append(_fake_sam_result())
        mask = tool.get_mask_from_instance(instance_id)
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)

    def test_round_robin_worker_selection(self):
        """Workers should be selected in round-robin order."""
        tool = _make_tool()
        fake_workers = [_make_fake_worker() for _ in range(3)]
        tool.execution_pool = fake_workers

        img = _make_dummy_image()
        for _ in range(6):
            iid, _ = asyncio.get_event_loop().run_until_complete(tool.create())
            asyncio.get_event_loop().run_until_complete(
                tool.execute(iid, parameters={"bbox": [10, 20, 30, 40]}, image_data=[img])
            )

        for w in fake_workers:
            assert w.execute.remote.call_count == 2

    def test_bbox_coordinate_normalization(self):
        """Verify 0-1000 coords are scaled to actual image pixels."""
        tool = _make_tool()
        w = _make_fake_worker()
        tool.execution_pool = [w]

        img = _make_dummy_image(200, 100)  # 200w x 100h
        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [500, 500, 1000, 1000]}, image_data=[img])
        )

        # worker.execute.remote should have received scaled coordinates
        call_kwargs = w.execute.remote.call_args
        actual_bbox = call_kwargs.kwargs.get("bounding_box") or call_kwargs[1].get("bounding_box")
        # 500/1000*200=100, 500/1000*100=50, 1000/1000*200=200, 1000/1000*100=100
        assert actual_bbox == pytest.approx([100.0, 50.0, 200.0, 100.0])


# ---------------------------------------------------------------------------
# Integration test – requires Ray + GPU + segment_anything + SAM checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not all([
        __import__("importlib").util.find_spec("ray"),
        __import__("importlib").util.find_spec("segment_anything"),
        __import__("importlib").util.find_spec("torch"),
    ]),
    reason="ray, segment_anything, or torch not installed",
)
class TestIntegration:
    """End-to-end test with real Ray workers and SAM model.

    Run explicitly: pytest tests/test_image_sam_tool.py::TestIntegration -v -s
    """

    MODEL_PATH = "/data/zhuhairui/sam_vit_h.pth"

    @pytest.fixture(autouse=True)
    def check_model(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(f"SAM checkpoint not found at {self.MODEL_PATH}")

    def test_end_to_end_segmentation(self):
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        from verl.tools.image_sam_tool import ImageSAMTool

        config = {
            "model_path": self.MODEL_PATH,
            "model_type": "vit_h",
            "num_workers": 1,
            "num_gpus_per_worker": 0.5,
            "rate_limit": 5,
        }
        tool = ImageSAMTool(config, _make_tool_schema())

        instance_id, _ = asyncio.get_event_loop().run_until_complete(tool.create())
        img = _make_dummy_image(640, 480)

        resp, reward, metrics = asyncio.get_event_loop().run_until_complete(
            tool.execute(instance_id, parameters={"bbox": [100, 100, 500, 400]}, image_data=[img])
        )

        assert metrics["success"] is True
        assert resp.image is not None
        print(f"\nIntegration test passed: score={metrics['score']:.4f}, "
              f"mask_area={metrics['mask_area']}, mask_ratio={metrics['mask_ratio']:.2%}")

        asyncio.get_event_loop().run_until_complete(tool.release(instance_id))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
