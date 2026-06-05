#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


ROOT_FALLBACKS = {
    "/jiangwenhao/zhuhairui": "/nfsdata4/zhuhairui",
}


def load_json(path: Path):
    if orjson is not None:
        return orjson.loads(path.read_bytes())

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None:
        path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(raw_path: str, repo_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    for src_prefix, dst_prefix in ROOT_FALLBACKS.items():
        if raw_path.startswith(src_prefix):
            fallback = Path(raw_path.replace(src_prefix, dst_prefix, 1))
            if fallback.exists():
                return fallback

    relative = repo_root / raw_path
    if relative.exists():
        return relative

    raise FileNotFoundError(f"Could not resolve dataset path: {raw_path}")


def load_dataset_attr(dataset_info_path: Path, dataset_name: str) -> dict:
    dataset_info = load_json(dataset_info_path)
    if dataset_name not in dataset_info:
        raise KeyError(f"Dataset `{dataset_name}` not found in {dataset_info_path}.")

    attr = dataset_info[dataset_name]
    if attr.get("formatting") != "sharegpt":
        raise ValueError(f"Dataset `{dataset_name}` is not sharegpt formatted.")

    columns = attr.get("columns", {})
    tags = attr.get("tags", {})
    return {
        "file_name": attr["file_name"],
        "messages_key": columns.get("messages", "conversations"),
        "role_tag": tags.get("role_tag", "from"),
        "content_tag": tags.get("content_tag", "value"),
        "user_tag": tags.get("user_tag", "human"),
        "assistant_tag": tags.get("assistant_tag", "gpt"),
        "observation_tag": tags.get("observation_tag", "observation"),
        "function_tag": tags.get("function_tag", "function_call"),
        "system_tag": tags.get("system_tag", "system"),
    }


def inspect_dataset(dataset_name: str, dataset_path: Path, attr: dict) -> tuple[list[dict], dict]:
    data = load_json(dataset_path)
    odd_tags = {attr["user_tag"], attr["observation_tag"]}
    even_tags = {attr["assistant_tag"], attr["function_tag"]}
    invalid_samples = []
    reason_counter = Counter()

    for index, sample in enumerate(data):
        messages = sample.get(attr["messages_key"], [])
        system_message = ""

        if not isinstance(messages, list):
            reason = "messages_not_list"
            invalid = {
                "dataset_name": dataset_name,
                "index": index,
                "reason": reason,
                "invalid_turn": None,
                "expected_roles": [],
                "actual_role": type(messages).__name__,
                "system_message": system_message,
                "original_message_count": None,
                "messages_after_system_count": None,
                "messages_after_system": [],
                "original_sample": sample,
            }
            invalid_samples.append(invalid)
            reason_counter[reason] += 1
            continue

        raw_messages = messages
        if raw_messages and raw_messages[0].get(attr["role_tag"]) == attr["system_tag"]:
            system_message = raw_messages[0].get(attr["content_tag"], "")
            messages = raw_messages[1:]

        reason = None
        invalid_turn = None
        expected_roles = []
        actual_role = None

        for turn_idx, message in enumerate(messages):
            role = message.get(attr["role_tag"])
            allowed = odd_tags if turn_idx % 2 == 0 else even_tags
            if role not in allowed:
                reason = "bad_role_seq"
                invalid_turn = turn_idx
                expected_roles = sorted(allowed)
                actual_role = role
                break

        if reason is None and len(messages) % 2 != 0:
            reason = "odd_message_count"
            invalid_turn = len(messages) - 1
            expected_roles = []
            actual_role = None

        if reason is None:
            continue

        normalized_messages = [
            {
                "role": message.get(attr["role_tag"]),
                "content": message.get(attr["content_tag"]),
            }
            for message in messages
        ]
        invalid = {
            "dataset_name": dataset_name,
            "index": index,
            "reason": reason,
            "invalid_turn": invalid_turn,
            "expected_roles": expected_roles,
            "actual_role": actual_role,
            "system_message": system_message,
            "original_message_count": len(raw_messages),
            "messages_after_system_count": len(messages),
            "messages_after_system": normalized_messages,
            "original_sample": sample,
        }
        invalid_samples.append(invalid)
        reason_key = reason if actual_role is None else f"{reason}:{invalid_turn}:{actual_role}"
        reason_counter[reason_key] += 1

    summary = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "total_samples": len(data),
        "invalid_samples": len(invalid_samples),
        "invalid_ratio": round(len(invalid_samples) / len(data), 6) if data else 0.0,
        "reason_breakdown": dict(reason_counter.most_common()),
    }
    return invalid_samples, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Extract invalid sharegpt samples using LLaMA-Factory rules.")
    parser.add_argument(
        "--dataset-info",
        default="data/dataset_info.json",
        help="Path to LLaMA-Factory dataset_info.json",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset name in dataset_info.json. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug/invalid_samples",
        help="Directory to write extracted invalid samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_info_path = (repo_root / args.dataset_info).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    all_summaries = []

    for dataset_name in args.dataset:
        attr = load_dataset_attr(dataset_info_path, dataset_name)
        dataset_path = resolve_path(attr["file_name"], repo_root)
        invalid_samples, summary = inspect_dataset(dataset_name, dataset_path, attr)

        dump_json(output_dir / f"{dataset_name}.invalid_samples.json", invalid_samples)
        dump_json(output_dir / f"{dataset_name}.summary.json", summary)
        all_summaries.append(summary)

        print(
            f"{dataset_name}: saved {summary['invalid_samples']} invalid samples "
            f"out of {summary['total_samples']} to {output_dir}"
        )

    dump_json(output_dir / "summary.json", all_summaries)
    print(f"Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
