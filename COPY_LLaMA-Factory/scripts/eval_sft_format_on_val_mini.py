#!/usr/bin/env python3
"""Evaluate SFT assistant reply format on val-mini.

This script runs generation on each sample's final assistant turn in
`val-mini.json` and checks whether the model output strictly follows:

<reason>...</reason><tool_call>{"name": ..., "arguments": {...}}</tool_call>

or:

<reason>...</reason><Terminate>

It writes a summary JSON plus a per-sample JSONL file for failure analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if TYPE_CHECKING:
    from llamafactory.chat import ChatModel


DEFAULT_DATASET = "/nfsdata4/zhuhairui/zhuhairui/data/smartagentV2/for-cluster/val-mini.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "debug" / "format_eval"
PATH_PREFIX_CANDIDATES = [
    ("/jiangwenhao/", "/nfsdata4/"),
    ("/nfsdata4/", "/jiangwenhao/"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate assistant response format on val-mini.")
    parser.add_argument("--model-name-or-path", help="Checkpoint or model path for inference.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET, help="Path to val-mini style dataset json.")
    parser.add_argument("--output-dir", help="Directory to save summary and per-sample outputs.")
    parser.add_argument("--template", default="qwen3_vl", help="LLaMA-Factory template name.")
    parser.add_argument("--infer-backend", default="huggingface", help="Inference backend.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for each generation.")
    parser.add_argument("--cutoff-len", type=int, default=16384, help="Context length for inference.")
    parser.add_argument("--image-max-pixels", type=int, default=2_000_000)
    parser.add_argument("--image-min-pixels", type=int, default=40_000)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling. Default is greedy.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Only evaluate the first N samples.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset start index.")
    parser.add_argument("--gold-only", action="store_true", help="Do not load model; validate gold assistant format only.")
    parser.add_argument("--verbose-failures", type=int, default=20, help="How many failures to print to stdout.")
    parser.add_argument("--progress-every", type=int, default=20, help="Write progress summary every N samples.")
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Write a smaller summary and only keep failures/exceptions in predictions.jsonl.",
    )
    return parser.parse_args()


def resolve_image_path(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    for src_prefix, dst_prefix in PATH_PREFIX_CANDIDATES:
        if path.startswith(src_prefix):
            swapped = Path(dst_prefix + path[len(src_prefix) :])
            if swapped.exists():
                return str(swapped)

    return path


def parse_target_type(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if stripped.endswith("<Terminate>"):
        return "terminate", None

    match = re.search(r"<tool_call>(.*?)</tool_call>", stripped, re.S)
    if not match:
        return "unknown", None

    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return "tool_call", None

    return "tool_call", payload.get("name")


def normalize_for_preview(text: str, limit: int = 500) -> str:
    compact = text.replace("\n", "\\n")
    return compact[:limit]


def strip_optional_think_prefix(text: str) -> tuple[str, bool]:
    stripped = text.lstrip()
    match = re.match(r"^<think>.*?</think>\s*", stripped, re.S)
    if match:
        return stripped[match.end() :], True
    return text, False


def analyze_response_format(text: str, allow_think_prefix: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_valid": False,
        "issues": [],
        "end_type": None,
        "reason_text": None,
        "tool_name": None,
        "tool_arguments": None,
        "tool_json_raw": None,
        "has_reason_tag": False,
        "non_empty_reason": False,
        "has_tool_call_tag": False,
        "tool_json_valid": False,
        "tool_name_valid": False,
        "tool_arguments_valid": False,
        "strict_tail_valid": False,
        "had_think_prefix": False,
    }
    stripped = text.strip()
    if not stripped:
        result["issues"].append("empty_response")
        return result

    if allow_think_prefix:
        stripped, had_think_prefix = strip_optional_think_prefix(stripped)
        result["had_think_prefix"] = had_think_prefix

    if not stripped.startswith("<reason>"):
        result["issues"].append("missing_reason_open")
        return result

    close_idx = stripped.find("</reason>")
    if close_idx == -1:
        result["issues"].append("missing_reason_close")
        return result

    result["has_reason_tag"] = True
    reason_text = stripped[len("<reason>") : close_idx]
    result["reason_text"] = reason_text
    if reason_text.strip():
        result["non_empty_reason"] = True
    else:
        result["issues"].append("empty_reason")

    tail = stripped[close_idx + len("</reason>") :].strip()
    if not tail:
        result["issues"].append("missing_tail")
        return result

    if tail == "<Terminate>":
        result["end_type"] = "terminate"
        result["strict_tail_valid"] = True
        result["is_valid"] = result["non_empty_reason"]
        return result

    if not tail.startswith("<tool_call>"):
        result["issues"].append("tail_is_not_tool_call_or_terminate")
        return result

    if not tail.endswith("</tool_call>"):
        result["issues"].append("missing_tool_call_close")
        return result

    if not re.fullmatch(r"<tool_call>.*</tool_call>", tail, re.S):
        result["issues"].append("extra_text_around_tool_call")
        return result

    result["has_tool_call_tag"] = True
    result["end_type"] = "tool_call"
    result["strict_tail_valid"] = True

    tool_json_raw = tail[len("<tool_call>") : -len("</tool_call>")].strip()
    result["tool_json_raw"] = tool_json_raw
    try:
        payload = json.loads(tool_json_raw)
    except json.JSONDecodeError:
        result["issues"].append("tool_call_json_invalid")
        return result

    result["tool_json_valid"] = True
    if not isinstance(payload, dict):
        result["issues"].append("tool_call_json_not_object")
        return result

    tool_name = payload.get("name")
    if isinstance(tool_name, str) and tool_name.strip():
        result["tool_name_valid"] = True
        result["tool_name"] = tool_name
    else:
        result["issues"].append("tool_name_invalid")

    tool_arguments = payload.get("arguments")
    if isinstance(tool_arguments, dict):
        result["tool_arguments_valid"] = True
        result["tool_arguments"] = tool_arguments
    else:
        result["issues"].append("tool_arguments_not_object")

    result["is_valid"] = (
        result["non_empty_reason"]
        and result["tool_json_valid"]
        and result["tool_name_valid"]
        and result["tool_arguments_valid"]
    )
    return result


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_name = "gold_only" if args.gold_only else Path(args.model_name_or_path).name
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output_dir = DEFAULT_OUTPUT_ROOT / f"{run_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset, got {type(data).__name__}")

    return data


def build_chat_input(sample: dict[str, Any]) -> tuple[str | None, list[dict[str, str]], list[str] | None, dict[str, Any]]:
    messages = sample["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each sample must end with an assistant message.")

    history = messages[:-1]
    system = None
    if history and history[0]["role"] == "system":
        system = history[0]["content"]
        history = history[1:]

    chat_messages = []
    for msg in history:
        role = msg["role"]
        if role == "tool":
            role = "observation"
        chat_messages.append({"role": role, "content": msg["content"]})
    images = [resolve_image_path(path) for path in sample.get("images", [])]
    return system, chat_messages, images or None, messages[-1]


def build_chat_model(args: argparse.Namespace) -> "ChatModel":
    from llamafactory.chat import ChatModel

    infer_args = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "infer_backend": args.infer_backend,
        "trust_remote_code": args.trust_remote_code,
        "cutoff_len": args.cutoff_len,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
    }
    return ChatModel(infer_args)


def update_counters(metrics: Counter[str], parsed: dict[str, Any], expected_type: str, expected_tool_name: str | None) -> None:
    metrics["samples"] += 1
    metrics[f"expected_{expected_type}"] += 1

    if parsed["is_valid"]:
        metrics["strict_format_pass"] += 1

    if parsed["has_reason_tag"]:
        metrics["has_reason_tag"] += 1
    if parsed["non_empty_reason"]:
        metrics["non_empty_reason"] += 1
    if parsed["has_tool_call_tag"]:
        metrics["has_tool_call_tag"] += 1
    if parsed["tool_json_valid"]:
        metrics["tool_json_valid"] += 1
    if parsed["tool_name_valid"]:
        metrics["tool_name_valid"] += 1
    if parsed["tool_arguments_valid"]:
        metrics["tool_arguments_valid"] += 1
    if parsed["strict_tail_valid"]:
        metrics["strict_tail_valid"] += 1
    if parsed["had_think_prefix"]:
        metrics["had_think_prefix"] += 1

    if parsed["end_type"] == expected_type:
        metrics["expected_end_type_match"] += 1

    if expected_type == "tool_call" and parsed["tool_name"] == expected_tool_name:
        metrics["expected_tool_name_match"] += 1

    if parsed["end_type"] == "terminate":
        metrics["predicted_terminate"] += 1
    elif parsed["end_type"] == "tool_call":
        metrics["predicted_tool_call"] += 1
    else:
        metrics["predicted_other"] += 1


def build_primary_summary(
    args: argparse.Namespace,
    end_index: int,
    metrics: Counter[str],
    issue_counter: Counter[str],
    expected_tool_counter: Counter[str],
    predicted_tool_counter: Counter[str],
    turn_counter: Counter[str],
    per_turn_metrics: dict[str, Counter[str]],
    predictions_path: Path,
    completed: bool,
) -> dict[str, Any]:
    total = max(metrics["samples"], 1)
    expected_tool_total = max(metrics["expected_tool_call"], 1)
    summary = {
        "dataset_path": str(args.dataset_path),
        "model_name_or_path": args.model_name_or_path,
        "gold_only": args.gold_only,
        "compact_output": args.compact_output,
        "completed": completed,
        "evaluated_samples": metrics["samples"],
        "sample_range": {"start_index": args.start_index, "end_index_exclusive": end_index},
        "primary_metrics": {
            "format_pass_rate": metrics["strict_format_pass_without_think"] / total,
            "end_type_match_rate": metrics["expected_end_type_match_without_think"] / total,
            "tool_name_match_rate": metrics["expected_tool_name_match_without_think"] / expected_tool_total,
            "think_prefix_rate": metrics["had_think_prefix"] / total,
            "generation_exception_rate": metrics["generation_exceptions"] / total,
        },
        "counts": {
            "samples": metrics["samples"],
            "expected_terminate": metrics["expected_terminate"],
            "expected_tool_call": metrics["expected_tool_call"],
            "format_pass": metrics["strict_format_pass_without_think"],
            "end_type_match": metrics["expected_end_type_match_without_think"],
            "tool_name_match": metrics["expected_tool_name_match_without_think"],
            "had_think_prefix": metrics["had_think_prefix"],
            "generation_exceptions": metrics["generation_exceptions"],
        },
        "artifacts": {
            "predictions_jsonl": str(predictions_path),
        },
    }
    if not args.compact_output:
        summary.update(
            {
                "issue_counts": dict(issue_counter.most_common()),
                "expected_tool_counts": dict(expected_tool_counter.most_common()),
                "predicted_tool_counts": dict(predicted_tool_counter.most_common()),
                "turn_distribution": dict(turn_counter),
                "per_turn_metrics": {
                    turn: {
                        "samples": counter["samples"],
                        "format_pass_rate": (
                            counter["strict_format_pass_without_think"] / counter["samples"] if counter["samples"] else 0.0
                        ),
                        "end_type_match_rate": (
                            counter["expected_end_type_match_without_think"] / counter["samples"]
                            if counter["samples"]
                            else 0.0
                        ),
                    }
                    for turn, counter in sorted(per_turn_metrics.items(), key=lambda item: item[0])
                },
            }
        )
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()


def main() -> None:
    args = parse_args()
    if not args.gold_only and not args.model_name_or_path:
        raise SystemExit("--model-name-or-path is required unless --gold-only is used.")

    dataset = load_dataset(args.dataset_path)
    end_index = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    samples = dataset[args.start_index:end_index]
    output_dir = build_output_dir(args)

    print(f"Dataset: {args.dataset_path}")
    print(f"Samples: {len(samples)} (from index {args.start_index} to {end_index - 1})")
    print(f"Output dir: {output_dir}")

    chat_model = None if args.gold_only else build_chat_model(args)

    metrics: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    expected_tool_counter: Counter[str] = Counter()
    predicted_tool_counter: Counter[str] = Counter()
    turn_counter: Counter[str] = Counter()
    per_turn_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    failures_to_print = args.verbose_failures

    predictions_path = output_dir / "predictions.jsonl"
    progress_summary_path = output_dir / "progress_summary.json"
    with predictions_path.open("w", encoding="utf-8") as writer:
        for local_index, sample in enumerate(samples):
            sample_index = args.start_index + local_index
            system, chat_messages, images, target_message = build_chat_input(sample)
            expected_text = target_message["content"]
            expected_type, expected_tool_name = parse_target_type(expected_text)
            extra_info = sample.get("extra_info") or {}
            turn_key = str(extra_info.get("turn", "unknown"))

            if expected_tool_name is not None:
                expected_tool_counter[expected_tool_name] += 1
            turn_counter[turn_key] += 1

            record: dict[str, Any] = {
                "sample_index": sample_index,
                "turn": extra_info.get("turn"),
                "total_turns": extra_info.get("total_turns"),
                "expected_type": expected_type,
                "expected_tool_name": expected_tool_name,
            }
            if not args.compact_output:
                record.update(
                    {
                        "history_roles": [msg["role"] for msg in chat_messages],
                        "image_count": len(images or []),
                        "gold_response": expected_text,
                    }
                )

            try:
                if args.gold_only:
                    predicted_text = expected_text
                    finish_reason = "gold_only"
                else:
                    responses = chat_model.chat(
                        messages=chat_messages,
                        system=system,
                        images=images,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                    )
                    predicted_text = responses[0].response_text
                    finish_reason = responses[0].finish_reason
                    record["response_length"] = responses[0].response_length
                    record["prompt_length"] = responses[0].prompt_length
                    record["finish_reason"] = finish_reason
                    metrics[f"finish_reason_{finish_reason}"] += 1
            except Exception as exc:  # noqa: BLE001
                metrics["generation_exceptions"] += 1
                issue_counter["generation_exception"] += 1
                record["status"] = "exception"
                record["exception"] = repr(exc)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                if failures_to_print > 0:
                    failures_to_print -= 1
                    print(f"[sample {sample_index}] generation exception: {exc}")
                continue

            parsed = analyze_response_format(predicted_text, allow_think_prefix=False)
            parsed_without_think = analyze_response_format(predicted_text, allow_think_prefix=True)
            update_counters(metrics, parsed, expected_type, expected_tool_name)
            if parsed_without_think["had_think_prefix"]:
                metrics["had_think_prefix"] += 1
            if parsed_without_think["is_valid"]:
                metrics["strict_format_pass_without_think"] += 1
            if parsed_without_think["end_type"] == expected_type:
                metrics["expected_end_type_match_without_think"] += 1
                per_turn_metrics[turn_key]["expected_end_type_match_without_think"] += 1
            if expected_type == "tool_call" and parsed_without_think["tool_name"] == expected_tool_name:
                metrics["expected_tool_name_match_without_think"] += 1
            per_turn_metrics[turn_key].update(
                {
                    "samples": 1,
                    "strict_format_pass": int(parsed["is_valid"]),
                    "strict_format_pass_without_think": int(parsed_without_think["is_valid"]),
                    "expected_end_type_match": int(parsed["end_type"] == expected_type),
                }
            )

            if parsed_without_think["tool_name"] is not None:
                predicted_tool_counter[parsed_without_think["tool_name"]] += 1

            for issue in parsed_without_think["issues"]:
                issue_counter[issue] += 1

            record.update(
                {
                    "status": "ok" if parsed_without_think["is_valid"] else "format_failure",
                    "predicted_response_preview": normalize_for_preview(predicted_text),
                    "predicted_end_type": parsed_without_think["end_type"],
                    "predicted_tool_name": parsed_without_think["tool_name"],
                    "issues": parsed_without_think["issues"],
                    "end_type_match": parsed_without_think["end_type"] == expected_type,
                    "tool_name_match": parsed_without_think["tool_name"] == expected_tool_name,
                }
            )
            if not args.compact_output:
                record.update(
                    {
                        "raw_status": "ok" if parsed["is_valid"] else "format_failure",
                        "predicted_response": predicted_text,
                        "parsed": parsed,
                        "parsed_without_think": parsed_without_think,
                    }
                )

            should_write_record = (not args.compact_output) or record["status"] != "ok"
            if should_write_record:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()

            if (not parsed_without_think["is_valid"] or parsed_without_think["end_type"] != expected_type) and failures_to_print > 0:
                failures_to_print -= 1
                print(
                    f"[sample {sample_index}] status={record['status']} expected={expected_type}/{expected_tool_name} "
                    f"pred={parsed_without_think['end_type']}/{parsed_without_think['tool_name']} "
                    f"issues={parsed_without_think['issues']}"
                )
                print(f"  preview: {record['predicted_response_preview']}")

            if (local_index + 1) % args.progress_every == 0:
                progress_summary = build_primary_summary(
                    args=args,
                    end_index=end_index,
                    metrics=metrics,
                    issue_counter=issue_counter,
                    expected_tool_counter=expected_tool_counter,
                    predicted_tool_counter=predicted_tool_counter,
                    turn_counter=turn_counter,
                    per_turn_metrics=per_turn_metrics,
                    predictions_path=predictions_path,
                    completed=False,
                )
                write_json(progress_summary_path, progress_summary)
                print(
                    f"Processed {local_index + 1}/{len(samples)} | "
                    f"format_pass_without_think={metrics['strict_format_pass_without_think']}/{metrics['samples']}"
                )

    summary = build_primary_summary(
        args=args,
        end_index=end_index,
        metrics=metrics,
        issue_counter=issue_counter,
        expected_tool_counter=expected_tool_counter,
        predicted_tool_counter=predicted_tool_counter,
        turn_counter=turn_counter,
        per_turn_metrics=per_turn_metrics,
        predictions_path=predictions_path,
        completed=True,
    )
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    print(f"Summary written to: {summary_path}")
    print(f"Predictions written to: {predictions_path}")
    print(json.dumps(summary["primary_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
