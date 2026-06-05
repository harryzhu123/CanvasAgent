#!/usr/bin/env python3
"""Generate preview user prompts and image prompts from compact stage12 task specs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


DEFAULT_INPUT = Path(__file__).resolve().parent / "rl_batch_2000_v5_long_chain_stage12_keypoints.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "qwen_preview_stage12_first10.json"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """You convert one compact multimodal task specification into structured dataset content.

Return strict JSON only, with exactly these two keys:
- userprompt: string
- image_prompt: array of strings

Requirements for userprompt:
- Write entirely in English.
- Make it sound natural, realistic, and like a direct request from an end user.
- Preserve the required dependency order and key constraints from the task.
- Do not mention tool names, JSON field names, implementation details, or dataset terminology.
- If the task needs input images, write as if the user has already provided those images.
- If the task needs multiple input images, naturally refer to "the first image", "the second image", and "the third image" where appropriate.

Requirements for image_prompt:
- Write entirely in English.
- If input_spec.mode_hint is prompt_only, return an empty list.
- Otherwise return exactly input_spec.image_count prompts.
- Each prompt must correspond one-to-one to the asset_roles item in the same position.
- Each prompt must be detailed enough for high-quality photorealistic image generation.
- Each prompt must follow the corresponding must_contain and must_avoid constraints.

Do not use markdown code fences. Do not add any explanation outside the JSON object."""

CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--sampling",
        choices=["sequential", "stratified_task_type"],
        default="sequential",
        help="How to select preview samples from the compact stage12 dataset.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Only keep records whose input_spec.image_count is greater than zero.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_records(records: list[dict], start: int, limit: int, sampling: str, require_images: bool) -> list[dict]:
    if require_images:
        records = [record for record in records if record.get("input_spec", {}).get("image_count", 0) > 0]

    if sampling == "sequential":
        return records[start : start + limit]

    groups: dict[str, list[dict]] = defaultdict(list)
    task_order: list[str] = []
    for record in records:
        task_type = record["task_type"]
        if task_type not in groups:
            task_order.append(task_type)
        groups[task_type].append(record)

    selected: list[dict] = []
    offset = start
    while len(selected) < limit:
        added_this_round = False
        for task_type in task_order:
            group = groups[task_type]
            if offset < len(group):
                selected.append(group[offset])
                added_this_round = True
                if len(selected) >= limit:
                    break
        if not added_this_round:
            break
        offset += 1
    return selected


def build_user_message(record: dict) -> str:
    compact_payload = {
        "sample_id": record["sample_id"],
        "task_type": record["task_type"],
        "cross_image": record["cross_image"],
        "scene": record["scene"],
        "task_brief": record["task_brief"],
        "key_variables": record["key_variables"],
        "constraints": record["constraints"],
        "style_requirements": record["style_requirements"],
        "input_spec": record["input_spec"],
        "output_spec": record["output_spec"],
    }
    return (
        "Transform the following compact task spec into final structured output. "
        "Return JSON only.\n\n"
        f"{json.dumps(compact_payload, ensure_ascii=False, indent=2)}"
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_response(text: str) -> dict:
    cleaned = strip_code_fences(text)
    return json.loads(cleaned)


def validate_output(record: dict, result: dict) -> list[str]:
    issues = []
    if not isinstance(result, dict):
        return ["response is not a JSON object"]

    expected_keys = {"userprompt", "image_prompt"}
    if set(result.keys()) != expected_keys:
        issues.append(f"keys mismatch: {sorted(result.keys())}")

    userprompt = result.get("userprompt")
    image_prompt = result.get("image_prompt")
    if not isinstance(userprompt, str) or not userprompt.strip():
        issues.append("userprompt missing or empty")
    elif CJK_PATTERN.search(userprompt):
        issues.append("userprompt contains CJK characters")
    if any(tool in (userprompt or "") for tool in ["ImageGeneration", "ImageEdit", "Grounding", "SAM", "OCR", "Crop", "Rotate", "Flip", "SR", "Overlayer", "Extract"]):
        issues.append("userprompt mentions tool names")

    if not isinstance(image_prompt, list):
        issues.append("image_prompt is not a list")
        return issues

    expected_count = record["input_spec"]["image_count"]
    if len(image_prompt) != expected_count:
        issues.append(f"image_prompt count mismatch: expected {expected_count}, got {len(image_prompt)}")

    for idx, prompt in enumerate(image_prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            issues.append(f"image_prompt[{idx}] missing or empty")
        elif CJK_PATTERN.search(prompt):
            issues.append(f"image_prompt[{idx}] contains CJK characters")

    return issues


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_one(client: OpenAI, model: str, record: dict, temperature: float) -> tuple[dict, str]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(record)},
        ],
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content.strip()
    return parse_response(raw_text), raw_text


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")

    records = load_records(args.input)
    selected = select_records(records, args.start, args.limit, args.sampling, args.require_images)
    client = create_client(api_key=api_key, base_url=args.base_url)

    outputs = []
    for record in selected:
        final_result = None
        final_raw = ""
        issues: list[str] = []
        for attempt in range(1, args.max_retries + 1):
            try:
                result, raw_text = generate_one(client, args.model, record, args.temperature)
                current_issues = validate_output(record, result)
                final_result = result
                final_raw = raw_text
                issues = current_issues
                if not current_issues:
                    break
            except Exception as exc:  # noqa: BLE001
                final_result = None
                final_raw = ""
                issues = [f"attempt {attempt} failed: {exc}"]
            time.sleep(1)

        outputs.append(
            {
                "sample_id": record["sample_id"],
                "task_type": record["task_type"],
                "required_tools": record.get("required_tools", []),
                "toolchainlength": record.get("toolchainlength"),
                "input_spec": record["input_spec"],
                "result": final_result,
                "issues": issues,
                "raw_response": final_raw,
            }
        )
        if args.flush_every > 0 and len(outputs) % args.flush_every == 0:
            args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    success_count = sum(1 for item in outputs if not item["issues"])
    print(f"saved={args.output}")
    print(f"total={len(outputs)} success={success_count} failed={len(outputs) - success_count}")


if __name__ == "__main__":
    main()
