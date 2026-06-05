#!/usr/bin/env python3
"""Rerun multiturn_reward metrics from saved trajectory directories."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import glob
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Keep reward logging quiet unless the caller explicitly asks otherwise.
os.environ.setdefault("REWARD_LOG_LEVEL", "quiet")
os.environ.setdefault("REWARD_DEBUG_JSONL", "0")

import multiturn_reward  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _clean_text(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", str(text))
    text = text.replace("<image>", "")
    return re.sub(r"\s+", " ", text).strip()[-2000:]


def _content_to_text(content: Any, image_marker: str = "[image]") -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "text":
                parts.append(str(item.get("text", "")))
            elif typ == "image":
                parts.append(image_marker)
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _extract_user_task_from_messages(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return _clean_text(_content_to_text(msg.get("content")))
    return ""


def _extract_user_task_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return ""
    last = ""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            last = _content_to_text(msg.get("content"))
    return _clean_text(last)


def _solution_from_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    seen_first_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            if not seen_first_user:
                seen_first_user = True
                continue
            text = _content_to_text(msg.get("content"))
            parts.append(f"user\n{text}")
        elif role == "tool":
            text = _content_to_text(msg.get("content"))
            parts.append(f"user\n<tool_response>\n{text}\n</tool_response>")
        elif role == "assistant":
            text = _content_to_text(msg.get("content"))
            parts.append(f"assistant\n{text}")
    return "\n".join(parts)


def _image_to_b64(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _find_image(sample_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = sample_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _final_image_path(sample_dir: Path, traj: dict) -> Path | None:
    keys = [k for k in traj.get("tool_output_keys", []) if not re.match(r"^img_\d+$", str(k))]
    for key in reversed(keys):
        path = _find_image(sample_dir, str(key))
        if path:
            return path
    return None


def _build_dataset_index(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    by_sample_index: dict[int, list[int]] = {}
    prompt_texts: dict[int, str] = {}
    for row_idx, row in df.iterrows():
        info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        sample_idx = info.get("index")
        if sample_idx is None:
            continue
        by_sample_index.setdefault(int(sample_idx), []).append(int(row_idx))
        prompt_texts[int(row_idx)] = _extract_user_task_from_prompt(row.get("prompt"))
    return df, by_sample_index, prompt_texts


def _match_row(
    traj: dict,
    task_text: str,
    df: pd.DataFrame,
    by_sample_index: dict[int, list[int]],
    prompt_texts: dict[int, str],
):
    sample_idx = int(traj.get("sample_index"))
    candidates = by_sample_index.get(sample_idx, [])
    if not candidates:
        raise KeyError(f"No parquet row found for sample_index={sample_idx}")
    scored = [(SequenceMatcher(None, task_text, prompt_texts[i]).ratio(), i) for i in candidates]
    scored.sort(reverse=True)
    best_ratio, row_idx = scored[0]
    row = df.iloc[row_idx]
    return int(row_idx), float(best_ratio), row


def _load_existing(path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            existing[str(record["sample"])] = record
    return existing


def _score_one(sample_dir: Path, df, by_sample_index, prompt_texts, step: int) -> dict:
    traj_path = sample_dir / "trajectory.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    messages = traj.get("messages", [])
    task_text = _extract_user_task_from_messages(messages)
    row_idx, match_ratio, row = _match_row(traj, task_text, df, by_sample_index, prompt_texts)

    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    ground_truth = reward_model.get("ground_truth")
    extra_info = _jsonable(row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {})
    extra_info.update(
        {
            "final_image_b64": _image_to_b64(_final_image_path(sample_dir, traj)),
            "input_image_b64": _image_to_b64(_find_image(sample_dir, "img_1")),
            "tool_rewards": traj.get("tool_rewards", []),
            "sample_id": traj.get("request_id") or sample_dir.name,
            "step": step,
        }
    )

    solution_str = _solution_from_messages(messages)
    reward_info = multiturn_reward.compute_score(
        data_source=str(row.get("data_source", "")),
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        step=step,
    )

    return {
        "sample": sample_dir.name,
        "sample_index": traj.get("sample_index"),
        "row_index": row_idx,
        "match_ratio": match_ratio,
        "data_source": str(row.get("data_source", "")),
        "gts": _jsonable(ground_truth),
        "task": task_text,
        **_jsonable(reward_info),
    }


def _mean(records: list[dict], key: str) -> float | None:
    vals = [float(r.get(key)) for r in records if isinstance(r.get(key), (int, float, bool))]
    return sum(vals) / len(vals) if vals else None


def _summarize(records: list[dict], eval_dir: Path) -> dict:
    fields = [
        "score",
        "judge_score",
        "image_prompt_judge_score",
        "aesthetic_judge_score",
        "trajectory_judge_score",
        "format_score",
        "action_process_score",
        "outcome_reward",
        "process_reward",
        "tool_call_count",
        "expected_tool_count",
        "expected_tool_path_length",
        "tool_coverage_ratio",
        "interaction_depth_ratio",
        "efficiency_penalty",
        "error_count",
        "api_failed",
        "image_prompt_api_failed",
        "aesthetic_api_failed",
        "trajectory_api_failed",
        "fatal_error",
    ]
    metrics = {f"{field}/mean": _mean(records, field) for field in fields}
    return {
        "eval_dir": str(eval_dir),
        "num_samples": len(records),
        "metrics": metrics,
        "table_row": {
            "reward": metrics["score/mean"],
            "img_judge": metrics["image_prompt_judge_score/mean"],
            "aes_judge": metrics["aesthetic_judge_score/mean"],
            "traj_judge": metrics["trajectory_judge_score/mean"],
            "process": metrics["process_reward/mean"],
            "tool_calls": metrics["tool_call_count/mean"],
            "expected": metrics["expected_tool_count/mean"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sample_dirs = sorted(
        [Path(p) for p in glob.glob(str(args.trace_dir / "sample_*_n0"))],
        key=lambda p: int(re.search(r"sample_(\d+)_n0$", p.name).group(1)),
    )
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]

    df, by_sample_index, prompt_texts = _build_dataset_index(args.parquet)
    existing = _load_existing(args.output_jsonl)
    todo = [p for p in sample_dirs if p.name not in existing]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"trace_dir={args.trace_dir}")
    print(f"samples={len(sample_dirs)} existing={len(existing)} todo={len(todo)} workers={args.workers}")

    mode = "a" if args.output_jsonl.exists() else "w"
    completed = 0
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_score_one, sample_dir, df, by_sample_index, prompt_texts, args.step): sample_dir
                for sample_dir in todo
            }
            for fut in as_completed(futures):
                sample_dir = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:
                    record = {"sample": sample_dir.name, "error": repr(exc)}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                existing[record["sample"]] = record
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    print(f"[{completed}/{len(todo)}] {record['sample']}")

    records = [r for r in existing.values() if "error" not in r]
    summary = _summarize(records, args.trace_dir)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    row = summary["table_row"]
    print("summary saved:", args.summary_json)
    print(
        "table:",
        f"reward={row['reward']:.4f}",
        f"img={row['img_judge']:.4f}",
        f"aes={row['aes_judge']:.4f}",
        f"traj={row['traj_judge']:.4f}",
        f"process={row['process']:.4f}",
        f"tools={row['tool_calls']:.4f}/{row['expected']:.4f}",
    )


if __name__ == "__main__":
    main()
