from verl.tools.utils.tool_registry import initialize_tools_from_config

tool_config_path = "examples/qwen3vl_multiturn/config/tool_config/image_tools_config_matched.yaml"
tools = initialize_tools_from_config(tool_config_path)
tool_dict = {tool.name: tool for tool in tools}

# 验证所有工具名称
expected_names = [
    "ImageGeneration", "Crop", "ImageEdit", "OCR", "Grounding",
    "Rotate", "Flip", "SAM", "SR", "Extract", "Overlayer"
]

for name in expected_names:
    assert name in tool_dict, f"Tool {name} not found!"
    print(f"✓ {name} found")

