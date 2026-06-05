#!/usr/bin/env python3
"""
Evaluation metrics for multi-turn visual tool-use agent.

Two judge calls per sample:
  S_inst  — Trajectory quality (1-10): task completion, image fidelity,
             visual awareness/self-correction, tool efficiency.
             Judge sees: task + input_img + trajectory + final_img.
  S_qual  — Aesthetic/visual quality (1-10): sharpness, color, composition,
             absence of artifacts.
             Judge sees: final_img only.
  CR      — Task Completion Rate (%)
  Avg.Calls — Average tool calls per trajectory

Usage:
  python eval_metrics.py --eval_dir <path/to/val_step_X> [--output results.json]
                          [--workers 8] [--skip_judge]
"""

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_BASE     = os.getenv("REWARD_API_BASE",    "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY      = os.getenv("REWARD_API_KEY", "")
MODEL_NAME   = os.getenv("REWARD_MODEL_NAME",  "qwen3.5-plus")
API_TIMEOUT  = int(os.getenv("REWARD_TIMEOUT",     "120"))
MAX_RETRIES  = int(os.getenv("REWARD_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Judge system prompts
# ---------------------------------------------------------------------------
INST_SYSTEM = """\
You are an expert evaluator of a visual AI agent that solves image editing and \
generation tasks by calling a fixed set of tools across multiple turns.

You will be given:
1. The task description (what the user asked for)
2. The user's input image (if the task involves editing an existing image)
3. The agent's full response trajectory (all reasoning steps and tool calls)
4. The final image produced by the agent (if any was produced)

## Available tools
ImageGeneration, ImageEdit, Crop, Rotate, Flip, SR, OCR, Grounding, SAM, Extract, Overlayer

## Evaluation dimensions

Score the agent across four dimensions:

1. **Task completion** (~50%): Did the agent fulfil the user's request? Are all \
requested edits/generations actually done? Are the results correct?

2. **Image fidelity** (~20%): Does the final image accurately match the task \
description? Are the requested elements, positions, colors, and text correct? \
For editing tasks, are unmodified regions properly preserved?

3. **Visual awareness & self-correction** (~20%): Does the agent demonstrate \
genuine visual understanding? After each tool returns an image, does the agent \
observe and describe what it sees? When the result does not match the user's \
request, does the agent identify the gap and retry with adjusted parameters? \
Reward accurate visual descriptions, correctly identifying mismatches, and \
purposeful retries. Penalise blindly assuming tools succeeded, claiming it \
cannot see images, or skipping verification before terminating.

4. **Tool efficiency** (~10%): Is the tool chain logical? Penalise hallucinated \
image IDs, completely irrelevant tool calls, or illogical tool ordering. \
A retry motivated by visual inspection is good self-correction, not waste.

## Scoring rubric (1-10)

10   Task fully completed. Image precisely matches the description. Agent \
demonstrated clear visual awareness and verified the result.
8-9  Task mostly completed with minor gaps. Image largely correct. Agent \
showed some visual awareness.
6-7  Partial progress. Some relevant tool calls but significant gaps remain. \
Agent showed limited visual awareness or skipped verification.
3-5  Little meaningful progress. Tool calls mostly irrelevant or incorrect. \
Agent showed no visual awareness.
1-2  No genuine attempt. Immediate termination without tool use, or output \
completely unrelated to the task, or no image produced when one was required.

## Important notes
- An agent that outputs only <Terminate> without calling any tools should score 1-2.
- If no image was produced and the task requires one, cap the score at 3.
- Penalise hallucinated image IDs.
- For editing tasks, heavily penalise destroying or ignoring the original content.
- An agent that claims "I cannot see images" should be penalised.

Output ONLY valid JSON with no markdown fences:
{"score": <integer 1-10>}"""


QUAL_SYSTEM = """\
You are an expert image quality evaluator.

You will be given a single image. Evaluate its aesthetic and visual quality \
purely on technical and perceptual grounds, ignoring what the user asked for.

Consider:
- Clarity and sharpness (no blur, no noise, no compression artifacts)
- Color naturalness and consistency (no color casts, no unnatural saturation)
- Composition and overall aesthetic appeal
- Absence of distortions, unnatural edges, visible seams, or hallucinated content
- Lighting consistency and realism

## Scoring rubric (1-10)

10   Excellent quality, professional-grade image, no perceptible flaws
8-9  High quality, minor imperfections not distracting
6-7  Acceptable quality, noticeable but not severe issues
4-5  Poor quality, significant artifacts or distortions
2-3  Very poor quality, heavily distorted or incoherent
1    Completely unusable / blank / corrupted / solid color

Output ONLY valid JSON with no markdown fences:
{"score": <integer 1-10>}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def _get_client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key=API_KEY)


def _parse_score(text: str | None, scale: int = 10) -> float | None:
    """Parse integer score from judge response."""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            v = float(data.get("score", -1))
            if 1 <= v <= scale:
                return v
        except Exception:
            pass
    m2 = re.search(r"\b([1-9]|10)\b", text)
    if m2:
        return float(m2.group())
    return None


def _call_judge(system: str, user_content: list, client, scale: int = 10) -> float | None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=32,
                timeout=API_TIMEOUT,
                extra_body={"enable_thinking": False},
            )
            return _parse_score(resp.choices[0].message.content, scale)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                print(f"[WARN] Judge failed after {MAX_RETRIES} retries: {e}")
    return None


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

def _extract_trajectory(messages: list) -> str:
    """Concatenate all assistant turn text content."""
    parts = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def load_sample(sample_dir: Path):
    traj_path = sample_dir / "trajectory.json"
    if not traj_path.exists():
        return None

    with open(traj_path) as f:
        traj = json.load(f)

    messages = traj.get("messages", [])

    # User question (task description)
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    if user_msg is None:
        return None
    content = user_msg["content"]
    question = next((c["text"] for c in content if c["type"] == "text"), "") if isinstance(content, list) else content

    # Trajectory text (all assistant turns)
    trajectory = _extract_trajectory(messages)

    # Input image
    input_img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = sample_dir / f"img_1{ext}"
        if p.exists():
            input_img_path = str(p)
            break

    # Tool output keys (exclude input images img_N)
    tool_keys = traj.get("tool_output_keys", [])
    call_keys = [k for k in tool_keys if not re.match(r"^img_\d+$", k)]
    tool_call_count = len(call_keys)

    # Final image: last tool-produced image file
    final_img_path = None
    for key in reversed(call_keys):
        for ext in (".jpg", ".jpeg", ".png"):
            p = sample_dir / f"{key}{ext}"
            if p.exists():
                final_img_path = str(p)
                break
        if final_img_path:
            break

    return {
        "sample":      sample_dir.name,
        "question":    question,
        "trajectory":  trajectory,
        "input_img":   input_img_path,
        "final_img":   final_img_path,
        "tool_calls":  tool_call_count,
    }


def score_sample(info: dict, client) -> dict:
    final_b64 = _encode_image(info["final_img"]) if info["final_img"] else None
    input_b64 = _encode_image(info["input_img"]) if info["input_img"] else None

    # --- S_inst: trajectory judge ---
    inst_content = [{"type": "text", "text": f"## Task Description\n{info['question']}"}]
    if input_b64:
        inst_content.append({"type": "text", "text": "\n## User Input Image"})
        inst_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_b64}"}})
    inst_content.append({"type": "text", "text": f"\n## Agent Trajectory\n{info['trajectory']}"})
    if final_b64:
        inst_content.append({"type": "text", "text": "\n## Final Output Image"})
        inst_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_b64}"}})
    else:
        inst_content.append({"type": "text", "text": "\n[No image was produced by the agent.]"})
    s_inst = _call_judge(INST_SYSTEM, inst_content, client)

    # --- S_qual: aesthetic quality, final image only ---
    s_qual = None
    if final_b64:
        qual_content = [
            {"type": "text", "text": "Please evaluate the aesthetic and visual quality of the following image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_b64}"}},
        ]
        s_qual = _call_judge(QUAL_SYSTEM, qual_content, client)

    return {
        "sample":     info["sample"],
        "completed":  s_inst is not None and s_inst >= 6,
        "tool_calls": info["tool_calls"],
        "s_inst":     s_inst,
        "s_qual":     s_qual,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",   required=True)
    parser.add_argument("--output",     default="")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--skip_judge", action="store_true")
    args = parser.parse_args()

    eval_dir    = Path(args.eval_dir)
    output_path = Path(args.output) if args.output else eval_dir / "metrics.json"

    sample_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sample_dirs)} samples in {eval_dir}")

    infos = [load_sample(d) for d in sample_dirs]
    infos = [i for i in infos if i]
    print(f"Loaded {len(infos)} valid trajectories")

    avg_calls = sum(i["tool_calls"] for i in infos) / len(infos) if infos else 0

    print(f"\n--- Quick metrics ---")
    print(f"Avg. Tool Calls: {avg_calls:.2f}")

    if args.skip_judge:
        summary = {"Avg_Calls": avg_calls, "n_total": len(infos)}
        with open(output_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved (no judge scores) → {output_path}")
        return

    print(f"\nStarting judge scoring with {args.workers} workers ...")
    client  = _get_client()
    results = []
    done    = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(score_sample, info, client): info for info in infos}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if done % 50 == 0 or done == len(infos):
                print(f"  [{done}/{len(infos)}] {res['sample']}  s_inst={res['s_inst']}  s_qual={res['s_qual']}")

    inst_vals  = [r["s_inst"] for r in results if r["s_inst"] is not None]
    qual_vals  = [r["s_qual"] for r in results if r["s_qual"] is not None]
    inst_mean  = sum(inst_vals) / len(inst_vals) if inst_vals else None
    qual_mean  = sum(qual_vals) / len(qual_vals) if qual_vals else None
    completed  = [r for r in results if r["completed"]]
    cr         = len(completed) / len(results) * 100 if results else 0

    print(f"\n{'='*45}")
    print(f"S_inst (Trajectory Quality, 1-10): {inst_mean:.2f}  (n={len(inst_vals)})")
    print(f"S_qual (Aesthetic Quality,  1-10): {qual_mean:.2f}  (n={len(qual_vals)})")
    print(f"CR (s_inst >= 6):                  {cr:.1f}%  ({len(completed)}/{len(results)})")
    print(f"Avg. Tool Calls:                   {avg_calls:.2f}")
    print(f"{'='*45}")

    output = {
        "summary": {
            "S_inst":        inst_mean,
            "S_qual":        qual_mean,
            "CR":            cr,
            "Avg_Calls":     avg_calls,
            "n_total":       len(infos),
            "n_completed":   len(completed),
            "n_scored_inst": len(inst_vals),
            "n_scored_qual": len(qual_vals),
        },
        "per_sample": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
