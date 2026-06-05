#!/usr/bin/env python3
"""
Convert RL-10K.json to verl-compatible parquet format.

Source format (RL-10K.json):
    {
        "path": ["Flip", "OCR", ...],
        "image": ["/path/to/image.png"],
        "prompt": "用户指令文本",
        "extra_info": {"id": 0, "tone": "explicit"}
    }

Target format (verl parquet):
    Columns: data_source, agent_name, prompt, ability, reward_model, extra_info, images

    - prompt: list of messages [{"role": "system", "content": "..."}, {"role": "user", "content": "<image>用户指令"}]
    - images: list of image file paths (for verl's rl_dataset.py to load via process_image)
    - reward_model: {"ground_truth": tool_path, "style": "rule"}
    - extra_info: {"index": id, "tone": ..., "image_paths": [...], "tool_path": [...]}

Key points:
    1. User message uses "<image>" placeholder so rl_dataset._build_messages() converts it to {"type": "image"}
    2. Image paths go into top-level "images" column (image_key="images") for rl_dataset to load
    3. System prompt from sys_prompt_reason.txt
"""

import json
import os

import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_JSON = "/data/zhuhairui/github-upload/RL_design/RL-10K.json"
OUTPUT_DIR = "/data/zhuhairui/data/rl10k"
SYSTEM_PROMPT_FILE = os.path.join(SCRIPT_DIR, "/nfsdata4/zhuhairui/zhuhairui/data/smartagentV2/for-41/sys_prompt_reason.txt")

TRAIN_SIZE = 9000


def load_system_prompt():
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        return f.read().strip()


def convert():
    with open(SOURCE_JSON, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {SOURCE_JSON}")

    system_prompt = load_system_prompt()

    records = []
    for item in data:
        item_id = item["extra_info"]["id"]
        tone = item["extra_info"].get("tone", "explicit")
        image_paths = item.get("image", [])
        tool_path = item.get("path", [])
        prompt_text = item["prompt"]

        # Build messages
        # If there are images, prepend <image> tag(s) to user message
        # so that rl_dataset._build_messages() can split them into {"type": "image"} entries
        if image_paths:
            image_tags = "<image>" * len(image_paths)
            user_content = f"{image_tags}{prompt_text}"
        else:
            user_content = prompt_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        record = {
            "data_source": "rl-10k",
            "agent_name": "tool_agent",
            "prompt": messages,
            "ability": "vision",
            "reward_model": {
                "ground_truth": tool_path,
                "style": "rule",
            },
            "extra_info": {
                "index": item_id,
                "tone": tone,
                "image_paths": image_paths,
                "tool_path": tool_path,
                "original_id": item_id,
            },
            # Top-level "images" column for rl_dataset to pick up (image_key="images")
            # Use None instead of empty list so rl_dataset skips image processing for no-image samples
            "images": [{"image": p} for p in image_paths] if image_paths else None,
        }
        records.append(record)

    # Split into train and test
    train_records = records[:TRAIN_SIZE]
    test_records = records[TRAIN_SIZE:]

    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Count stats
    train_with_img = sum(1 for r in train_records if r["images"])
    test_with_img = sum(1 for r in test_records if r["images"])
    print(f"Train with images: {train_with_img}/{len(train_records)}")
    print(f"Test with images: {test_with_img}/{len(test_records)}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)")

    # Verify
    print("\n--- Verification ---")
    verify_df = pd.read_parquet(train_path)
    print(f"Columns: {verify_df.columns.tolist()}")

    # Show a sample with image
    for i, row in verify_df.iterrows():
        if row["images"] is not None:
            print(f"\nSample {i} (with image):")
            print(f"  images: {row['images']}")
            print(f"  user msg: {row['prompt'][1]['content'][:120]}...")
            print(f"  tool_path: {row['reward_model']['ground_truth']}")
            break

    # Show a sample without image
    for i, row in verify_df.iterrows():
        if row["images"] is None:
            print(f"\nSample {i} (no image):")
            print(f"  images: {row['images']}")
            print(f"  user msg: {row['prompt'][1]['content'][:120]}...")
            print(f"  tool_path: {row['reward_model']['ground_truth']}")
            break


if __name__ == "__main__":
    convert()
