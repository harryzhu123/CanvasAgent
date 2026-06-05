#!/usr/bin/env python3
"""
Hybrid reward function for multi-turn visual tool-use RL training.

The reward is organised into four layers:
  - Outcome reward:
      * image_prompt_judge
      * aesthetic_judge
  - Process reward:
      * trajectory_judge
      * format_score
      * action_process_score
      * interaction_reward
  - Efficiency penalty:
      * error_penalty
      * repeat_penalty
      * length_penalty
      * tool_cost_penalty
  - Fatal error gate:
      * sets reward to zero for catastrophic failures

Configuration via environment variables:
  REWARD_API_BASE    — OpenAI-compatible API base (default: dashscope)
  REWARD_API_KEY     — API key (required)
  REWARD_MODEL_NAME  — judge model name (default: qwen3.5-plus)
  REWARD_TIMEOUT     — API call timeout in seconds (default: 180)
  REWARD_MAX_RETRIES — number of retry attempts on failure (default: 2)
  REWARD_FALLBACK    — score returned when a judge call fails (default: 0.3)
  REWARD_LOG_LEVEL   — quiet|summary|sample|debug (default: sample)
  REWARD_DEBUG_DIR   — directory for reward/judge jsonl debug files
"""

from collections import Counter
import hashlib
import json
import logging
import numbers
import os
import re
import time

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("REWARD_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("REWARD_API_KEY", "")
MODEL_NAME = os.getenv("REWARD_MODEL_NAME", "qwen3.5-plus")
API_TIMEOUT = int(os.getenv("REWARD_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("REWARD_MAX_RETRIES", "2"))
FALLBACK_SCORE = float(os.getenv("REWARD_FALLBACK", "0.3"))

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
REWARD_LOG_LEVEL = os.getenv("REWARD_LOG_LEVEL", "sample").strip().lower()
REWARD_DEBUG_DIR = os.getenv("REWARD_DEBUG_DIR", os.path.join(os.path.dirname(__file__), "log"))
REWARD_DEBUG_JSONL = os.getenv("REWARD_DEBUG_JSONL", "1").strip().lower() not in {"0", "false", "no", "off"}
REWARD_JUDGE_PREVIEW_CHARS = int(os.getenv("REWARD_JUDGE_PREVIEW_CHARS", "120"))
REWARD_SUMMARY_EVERY = int(os.getenv("REWARD_SUMMARY_EVERY", "16"))

_LOG_LEVELS = {
    "quiet": 0,
    "summary": 1,
    "sample": 2,
    "debug": 3,
}
_REWARD_WINDOW_STATS = {
    "n": 0,
    "scores": [],
    "zero": 0,
    "fatal": 0,
    "tool_errors": 0,
    "judge_fail": Counter(),
    "fatal_reasons": Counter(),
    "invalid_tools": Counter(),
}

# ---------------------------------------------------------------------------
# Reward weights and overall normalization
# ---------------------------------------------------------------------------
WEIGHT_IMAGE_PROMPT_JUDGE = float(os.getenv("REWARD_WEIGHT_IMAGE_PROMPT_JUDGE", "0.30"))
WEIGHT_AESTHETIC_JUDGE = float(os.getenv("REWARD_WEIGHT_AESTHETIC_JUDGE", "0.10"))
WEIGHT_TRAJECTORY_JUDGE = float(os.getenv("REWARD_WEIGHT_TRAJECTORY_JUDGE", "0.30"))
WEIGHT_FORMAT = float(os.getenv("REWARD_WEIGHT_FORMAT", "0.15"))
WEIGHT_ACTION_PROCESS = float(os.getenv("REWARD_WEIGHT_ACTION_PROCESS", "0.15"))
TOOL_COVERAGE_REWARD_MAX = float(os.getenv("REWARD_TOOL_COVERAGE_REWARD_MAX", "0.10"))
INTERACTION_REWARD_MAX = float(os.getenv("REWARD_INTERACTION_REWARD_MAX", "0.22"))
MISSING_KEY_TOOL_PENALTY_MAX = float(os.getenv("REWARD_MISSING_KEY_TOOL_PENALTY_MAX", "0.15"))

# Keep the final normalized reward on a fixed [0, 1] scale. Positive bonuses
# such as tool coverage / interaction depth can still help offset penalties,
# but we do not expand the normalization ceiling beyond 1.0.
TOTAL_REWARD_POSITIVE_MAX = float(os.getenv("REWARD_TOTAL_POSITIVE_MAX", "1.0"))
OUTCOME_REWARD_COMPONENT_MAX = WEIGHT_IMAGE_PROMPT_JUDGE + WEIGHT_AESTHETIC_JUDGE
PROCESS_REWARD_COMPONENT_MAX = (
    WEIGHT_TRAJECTORY_JUDGE + WEIGHT_FORMAT + WEIGHT_ACTION_PROCESS + TOOL_COVERAGE_REWARD_MAX + INTERACTION_REWARD_MAX
)
OUTCOME_REWARD_BUDGET = min(TOTAL_REWARD_POSITIVE_MAX, OUTCOME_REWARD_COMPONENT_MAX)
PROCESS_REWARD_BUDGET = max(TOTAL_REWARD_POSITIVE_MAX - OUTCOME_REWARD_BUDGET, 0.0)

TURN_TOKEN_LIMIT = int(os.getenv("REWARD_TURN_TOKEN_LIMIT", "256"))

VALID_TOOLS = {
    "ImageGeneration",
    "ImageEdit",
    "Crop",
    "Rotate",
    "Flip",
    "SR",
    "OCR",
    "Grounding",
    "SAM",
    "Extract",
    "Overlayer",
}

REQUIRED_ARGS = {
    "ImageGeneration": ("prompt",),
    "ImageEdit": ("edit_prompt",),
    "Crop": ("bbox",),
    "Rotate": ("angle",),
    "Flip": (),
    "SR": (),
    "OCR": (),
    "Grounding": ("reference_text",),
    "SAM": ("bbox",),
    "Extract": ("mask",),
    "Overlayer": ("overlay_type", "content"),
}

IMAGE_OUTPUT_TOOLS = {
    "ImageGeneration",
    "ImageEdit",
    "Crop",
    "Rotate",
    "Flip",
    "SAM",
    "SR",
    "Extract",
    "Overlayer",
}

IMAGE_TASK_TOOLS = IMAGE_OUTPUT_TOOLS - {"SAM"}
IMAGE_REF_KEYS = {"image", "image_ref", "base_image", "masked_image"}


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------
IMAGE_PROMPT_JUDGE_SYSTEM = """\
You are an expert evaluator for visual AI tasks.

You will be given:
1. The user's task prompt
2. The input image, if one was provided
3. The final output image, if one was produced
4. The final text response/trajectory only as auxiliary context

Score ONLY whether the final output satisfies the user's visual request.
Focus on semantic correctness, requested objects/actions, positions, colors,
text, preservation of the input image when editing, and overall visual fidelity.
Do not reward a good-looking process if the final output is wrong.

Be conservative with high scores:
- A score >= 0.9 is allowed ONLY when every major requested constraint is
  satisfied and almost all minor constraints are also correct, with no clearly
  wrong object, identity, text, count, spatial relation, edit target, or major
  omission.
- If the task requires visible text, then any misspelled word, wrong number,
  missing required text, garbled lettering, or clearly incorrect typography
  should usually cap the score at <= 0.3, and large text errors can justify
  scores near 0.0 even if the rest of the image looks good.
- If the final image follows only the broad theme of the prompt but misses one
  or more explicit requested constraints, such as the wrong object, wrong
  color, wrong pose, wrong count, wrong location, wrong style, or incomplete
  edit target, the score should usually be <= 0.4.
- For editing tasks, 0.9+ additionally requires the untouched regions to remain
  natural and the requested region to be edited precisely without damaging
  nearby content.
- If any major requested element is missing, added incorrectly, placed wrongly,
  uses the wrong text, changes the wrong region, or preserves the wrong object,
  the score should usually be <= 0.5.
- If the image is generally plausible but only captures the broad theme while
  missing specific requested constraints, keep the score in the 0.3-0.6 range.
- If the final image looks nice but is semantically wrong, still score it low.
- Do not infer success from the assistant's explanation alone; judge the image
  itself. If uncertain between two bands, choose the lower band.

Use this scale:
1.0 = fully satisfies all important requested constraints
0.9 = correct in all major aspects with only tiny non-critical flaws
0.7-0.8 = clearly strong result, but still missing some secondary details or has noticeable minor mistakes
0.4-0.6 = partial success; important requested requirements are missing or wrong
0.1-0.3 = barely related, weak attempt, or major semantic mismatch
0.0 = unrelated, empty, or no meaningful attempt

Output ONLY valid JSON with no markdown fences:
{"score": 0.0}"""


AESTHETIC_JUDGE_SYSTEM = """\
You are an expert evaluator of visual aesthetic quality for image generation and image editing tasks.

You will be given:
1. The user's task prompt
2. The user's input image, if one was provided
3. The final output image, if one was produced

Your job is to score ONLY the aesthetic quality of the final output image.
Do NOT score whether the image semantically satisfies the user's request.
Do NOT score whether the tool-use trajectory was reasonable.
Those are evaluated separately.

Focus only on visual quality, including:
- composition and balance
- color harmony and lighting consistency
- sharpness and clarity
- naturalness of edges, textures, and object boundaries
- realism or stylistic coherence
- absence of obvious artifacts, distortions, blur, ghosting, broken anatomy, pasted-looking objects, jagged masks, or mismatched shadows

Special guidance for editing tasks:
- Reward outputs that preserve the original image naturally while integrating edits cleanly.
- Penalize outputs with obvious edit seams, inconsistent perspective, color mismatch, unnatural blending, or damaged untouched regions.
- If visible text in the image is malformed, misspelled, broken, inconsistent,
  or obviously pasted in an unnatural way, do not give a high score even if the
  rest of the image is visually appealing.

Scoring rubric (0.0 to 1.0):
1.0   Visually polished and aesthetically strong. Clean composition, coherent style, natural blending, sharp details, no visible artifacts, and no visibly broken text.
0.7-0.9  Generally appealing and coherent, with only minor visual flaws and no major artifact or broken-text issue.
0.4-0.6  Mixed quality. Some parts look acceptable, but artifacts, weak composition, blur, broken text, or inconsistency are clearly noticeable.
0.1-0.3  Poor visual quality. Strong artifacts, unnatural blending, awkward layout, visibly bad text rendering, or severe degradation.
0.0   No usable final image, or the output is visually broken.

Important notes:
- If no final image is produced, score 0.0.
- Ignore whether the requested content was correct; judge aesthetics only.
- A semantically correct but ugly image should score low here.
- A visually pleasing but semantically wrong image can still score high here, because semantic correctness is evaluated elsewhere.

Output ONLY valid JSON with no markdown fences:
{"score": 0.0}"""


TRAJECTORY_JUDGE_SYSTEM = """\
You are an expert evaluator for a visual tool-use agent.

You will be given the user's task prompt, the expected tool set when available,
the agent's full trajectory, a parsed tool-call summary, and tool error counts.

Score ONLY whether the process is reasonable. Do not judge final image quality.
Reward logical tool selection, valid dependency handling, using outputs from
previous tools correctly, concise but sufficient reasoning, and appropriate
verification before final termination. Penalize irrelevant tools, hallucinated
image IDs, invalid dependencies, blind repetition, and unsupported claims.

Order is not mandatory if the same goal can be achieved another way, but all
necessary tools should appear when the expected tool set is provided.

Hard minimum-score rules:
- If the parsed tool-call summary contains exactly one tool call, output
  {"score": 0.0}. This is the lowest score and has no exceptions.
- A one-tool-call trajectory is considered an invalid process even if the final
  image appears good, because it did not demonstrate verification, reassessment,
  refinement, or meaningful multi-step tool use.
- Do not give partial process credit for a trajectory that only calls
  ImageGeneration once, ImageEdit once, or any other single tool once and then
  terminates.

Be conservative with high scores:
- A score >= 0.9 is allowed ONLY when the trajectory covers the key expected
  tools when applicable, respects dependencies, avoids obvious redundancy, and
  does not skip critical steps.
- A score >= 0.9 additionally requires that the agent responds sensibly to tool
  observations and errors, uses currently valid image IDs, and terminates only
  after there is strong evidence the task is complete.
- If the prompt explicitly asks for verification, reassessment, refinement, or
  iterative correction, then stopping after a single weak attempt without a real
  follow-up check should usually cap the score at <= 0.5.
- If expected tools are provided and multiple key tools are missing, the score
  should usually be <= 0.5 even if the visible process looks superficially neat.
- If the agent uses far fewer tools than the task appears to require, treat that
  as a substantial process defect rather than a small inefficiency.
- If the agent replaces several required deterministic tools with one vague or
  shortcut call, or declares success before all explicit prompt requirements are
  credibly addressed, the score should usually be <= 0.4.
- If the agent hallucinates success after a tool error, ignores a failed tool
  call, or claims completion without enough supporting intermediate evidence,
  the score should usually be <= 0.3.
- Repeated or unnecessary calls, unsupported claims, or weak verification should
  cap the score below the top band.
- If uncertain between two score bands, choose the lower one.

Use this scale:
1.0 = complete, dependency-consistent, efficient trajectory with clear justification
0.9 = all key steps are present and valid, with at most tiny non-critical inefficiency
0.7-0.8 = mostly reasonable, but has some unnecessary calls or small process gaps
0.4-0.6 = partially reasonable, but missing important tools/steps or showing clear process defects
0.1-0.3 = largely flawed, strongly incomplete, or tool usage is mostly incoherent
0.0 = no meaningful process or unusable tool trajectory

Output ONLY valid JSON with no markdown fences:
{"score": 0.0}"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    return _openai_client


def _log_enabled(level: str) -> bool:
    configured = _LOG_LEVELS.get(REWARD_LOG_LEVEL, _LOG_LEVELS["sample"])
    requested = _LOG_LEVELS.get(level, _LOG_LEVELS["sample"])
    return configured >= requested


def _preview(text: str | None, limit: int = REWARD_JUDGE_PREVIEW_CHARS) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned[:limit]


def _task_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:10]


def _first_non_empty(*values, default="na"):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _write_debug_jsonl(name: str, payload: dict):
    if not REWARD_DEBUG_JSONL:
        return
    try:
        os.makedirs(REWARD_DEBUG_DIR, exist_ok=True)
        path = os.path.join(REWARD_DEBUG_DIR, f"{name}_{time.strftime('%Y%m%d')}_{os.getpid()}.jsonl")
        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            **payload,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("Failed to write %s debug jsonl: %s", name, exc)


def _format_counter(counter: Counter, limit: int = 5) -> str:
    if not counter:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(limit))


# ---------------------------------------------------------------------------
# Judge calls with retry
# ---------------------------------------------------------------------------

def _call_judge(system_prompt: str, user_content: list[dict], label: str) -> float | None:
    """Call a judge model. Returns score or None."""
    for attempt in range(MAX_RETRIES + 1):
        start = time.monotonic()
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=64,
                timeout=API_TIMEOUT,
                extra_body={"enable_thinking": False},
            )
            raw = response.choices[0].message.content
            score, parse_method = _parse_score_with_method(raw)
            latency_ms = int((time.monotonic() - start) * 1000)
            ok = score is not None
            if _log_enabled("debug") or (not ok and _log_enabled("summary")):
                fallback = FALLBACK_SCORE if not ok else "na"
                print(
                    f"[JUDGE] label={label} ok={int(ok)} score={score} fallback={fallback} "
                    f"parse={parse_method} latency_ms={latency_ms} raw_len={len(raw or '')} "
                    f"preview={json.dumps(_preview(raw), ensure_ascii=False)}",
                    flush=True,
                )
            if _log_enabled("debug") or not ok:
                _write_debug_jsonl(
                    "judge_debug",
                    {
                        "label": label,
                        "ok": ok,
                        "score": score,
                        "parse": parse_method,
                        "latency_ms": latency_ms,
                        "raw": raw,
                    },
                )
            return score
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                if _log_enabled("summary"):
                    print(
                        f"[JUDGE] label={label} ok=0 parse=exception attempt={attempt + 1}/{MAX_RETRIES} "
                        f"retry_wait_s={wait} error={json.dumps(str(e)[:160], ensure_ascii=False)}",
                        flush=True,
                    )
                time.sleep(wait)
            else:
                logger.warning("Judge %s failed after %d retries, using fallback %.1f: %s",
                               label, MAX_RETRIES, FALLBACK_SCORE, e)
                _write_debug_jsonl(
                    "judge_debug",
                    {
                        "label": label,
                        "ok": False,
                        "parse": "exception",
                        "error": str(e),
                    },
                )
    return None


def _call_image_prompt_judge(
    task: str,
    solution_str: str,
    image_b64: str | None,
    input_image_b64: str | None = None,
) -> float | None:
    """Judge whether the final output image satisfies the prompt."""
    user_content = [{"type": "text", "text": f"## User Task Prompt\n{task}"}]
    if input_image_b64:
        user_content.append({"type": "text", "text": "\n## User Input Image"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{input_image_b64}"},
        })
    if image_b64:
        user_content.append({"type": "text", "text": "\n## Final Output Image"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    else:
        user_content.append({
            "type": "text",
            "text": "\n[No image was produced by the agent.]",
        })
    user_content.append({"type": "text", "text": f"\n## Final Text / Auxiliary Trajectory Context\n{solution_str[-4000:]}"})
    return _call_judge(IMAGE_PROMPT_JUDGE_SYSTEM, user_content, "image_prompt")


def _call_aesthetic_judge(
    task: str,
    image_b64: str | None,
    input_image_b64: str | None = None,
) -> float | None:
    """Judge only the aesthetic quality of the final output image."""
    if not image_b64:
        return 0.0

    user_content = [{"type": "text", "text": f"## User Task Prompt\n{task}"}]
    if input_image_b64:
        user_content.append({"type": "text", "text": "\n## User Input Image"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{input_image_b64}"},
        })
    user_content.append({"type": "text", "text": "\n## Final Output Image"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    })
    return _call_judge(AESTHETIC_JUDGE_SYSTEM, user_content, "aesthetic")


def _call_trajectory_judge(
    task: str,
    solution_str: str,
    expected_tools: list[str],
    parsed_tool_calls: list[dict],
    error_count: int,
) -> float | None:
    """Judge whether the tool-use trajectory is reasonable."""
    summary = [
        {
            "index": i,
            "name": call.get("name"),
            "arguments": call.get("arguments"),
        }
        for i, call in enumerate(parsed_tool_calls)
    ]
    user_content = [
        {"type": "text", "text": f"## User Task Prompt\n{task}"},
        {"type": "text", "text": f"\n## Expected Tools (order not mandatory)\n{expected_tools or 'unknown'}"},
        {"type": "text", "text": f"\n## Tool Error Count\n{error_count}"},
        {"type": "text", "text": f"\n## Tool Call Count\n{len(parsed_tool_calls)}"},
        {"type": "text", "text": f"\n## Parsed Tool Calls\n{json.dumps(summary, ensure_ascii=False)}"},
        {"type": "text", "text": f"\n## Agent Trajectory\n{solution_str[-8000:]}"},
    ]
    return _call_judge(TRAJECTORY_JUDGE_SYSTEM, user_content, "trajectory")


def _parse_score_with_method(text: str | None) -> tuple[float | None, str]:
    """Extract score from judge response and report the parse path."""
    if not text:
        return None, "empty"
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            v = float(data.get("score", -1))
            if 0.0 <= v <= 1.0:
                return v, "json_fragment"
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Regex fallback
    m = re.search(
        r"\bscore\b\s*[\"']?\s*(?:is|=|:)\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        v = float(m.group(1))
        return max(0.0, min(v if v <= 1.0 else v / 10.0, 1.0)), "regex_score"
    return None, "failed"


def _parse_score(text: str | None) -> float | None:
    """Extract score from judge response. Returns None if unparseable."""
    return _parse_score_with_method(text)[0]


# ---------------------------------------------------------------------------
# Deterministic reward helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)
_IMAGE_ID_RE = re.compile(
    r"^(?:img_\d+|base|base_image|ImageGeneration_\d+|ImageEdit_\d+|Crop_\d+|"
    r"Rotate_\d+|Flip_\d+|SR_\d+|SAM_\d+|Extract_\d+|Overlayer_\d+)$"
)
_IMAGE_ID_TOKEN_RE = re.compile(
    r"\b(?:img_\d+|base|base_image|ImageGeneration_\d+|ImageEdit_\d+|Crop_\d+|"
    r"Rotate_\d+|Flip_\d+|SR_\d+|SAM_\d+|Extract_\d+|Overlayer_\d+)\b"
)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _normalize_total_reward(raw_score: float) -> float:
    if TOTAL_REWARD_POSITIVE_MAX <= 1e-6:
        return _clamp01(raw_score)
    return _clamp01(raw_score / TOTAL_REWARD_POSITIVE_MAX)


def _canonical_tool_name(value) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    aliases = {
        "Generation": "ImageGeneration",
        "Gen": "ImageGeneration",
        "Edit": "ImageEdit",
        "SuperResolution": "SR",
        "Super-Resolution": "SR",
        "Segment": "SAM",
        "Segmentation": "SAM",
        "Overlay": "Overlayer",
    }
    name = aliases.get(name, name)
    return name if name in VALID_TOOLS else None


def _flatten_tool_spec(value) -> list[str]:
    """Extract canonical tool names from list/string specs."""
    if value is None:
        return []
    if np is not None and isinstance(value, np.ndarray):
        return _flatten_tool_spec(value.tolist())
    if isinstance(value, (list, tuple, set)):
        tools = []
        for item in value:
            tools.extend(_flatten_tool_spec(item))
        return tools
    text = str(value)
    parts = re.split(r"\s*(?:->|\+|,|/|;|\||，|、)\s*", text)
    tools = []
    for part in parts:
        tool = _canonical_tool_name(part)
        if tool:
            tools.append(tool)
    return tools


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_expected_tool_path(ground_truth, extra_info: dict) -> list[str]:
    candidates = []
    for key in ("tools", "tool_path", "required_tools", "expected_tools", "tool"):
        if key in extra_info:
            candidates.extend(_flatten_tool_spec(extra_info.get(key)))
    if not candidates:
        candidates.extend(_flatten_tool_spec(ground_truth))
    return candidates


def _extract_expected_tools(ground_truth, extra_info: dict) -> list[str]:
    return _unique_preserve_order(_extract_expected_tool_path(ground_truth, extra_info))


def _summarise_expected_tool_coverage(calls: list[dict], expected_tools: list[str]) -> dict:
    called_tools = []
    for call in calls:
        name = _canonical_tool_name(call.get("name"))
        if name:
            called_tools.append(name)
    called_unique_tools = _unique_preserve_order(called_tools)

    if not expected_tools:
        return {
            "coverage_ratio": 1.0,
            "expected_tool_count": 0,
            "called_unique_tool_count": len(called_unique_tools),
            "matched_expected_tools": [],
            "missing_expected_tools": [],
            "called_unique_tools": called_unique_tools,
        }

    matched_expected_tools = [tool for tool in expected_tools if tool in called_unique_tools]
    missing_expected_tools = [tool for tool in expected_tools if tool not in called_unique_tools]
    coverage_ratio = len(matched_expected_tools) / max(len(expected_tools), 1)
    return {
        "coverage_ratio": coverage_ratio,
        "expected_tool_count": len(expected_tools),
        "called_unique_tool_count": len(called_unique_tools),
        "matched_expected_tools": matched_expected_tools,
        "missing_expected_tools": missing_expected_tools,
        "called_unique_tools": called_unique_tools,
    }


def _extract_task_text(ground_truth, extra_info: dict) -> str:
    """Prefer the user prompt over reward_model.ground_truth when available."""
    for key in ("user_prompt", "task_prompt", "task", "prompt_text"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    prompt_str = extra_info.get("prompt_str")
    if isinstance(prompt_str, str) and prompt_str.strip():
        # Qwen-style chat templates often preserve role labels in decoded text.
        patterns = [
            r"<\|im_start\|>user\s*(.*?)<\|im_end\|>",
            r"(?:^|\n)user\s*\n(.*?)(?:\nassistant\s*\n|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, prompt_str, re.DOTALL | re.IGNORECASE)
            if matches:
                text = matches[-1].strip()
                if text:
                    return _clean_task_text(text)
        # Fall back to the last visible image marker and following user text.
        marker_matches = list(re.finditer(r"(?:<image>|Picture\s+\d+\s*:)", prompt_str, re.IGNORECASE))
        if marker_matches:
            text = prompt_str[marker_matches[-1].end():].strip()
            text = re.split(r"(?:assistant|<\|im_start\|>assistant)", text, flags=re.IGNORECASE)[0]
            if text.strip():
                return _clean_task_text(text)

    if isinstance(ground_truth, (list, tuple)):
        tools = _flatten_tool_spec(ground_truth)
        if tools and len(tools) == len(ground_truth):
            return "Expected tool path: " + " -> ".join(tools)
        return " -> ".join(str(t) for t in ground_truth)
    return str(ground_truth)


def _clean_task_text(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("<image>", "").strip()
    return text[-2000:]


def _parse_tool_calls(solution_str: str) -> tuple[list[dict], list[str]]:
    calls = []
    errors = []
    for match in _TOOL_CALL_RE.finditer(solution_str or ""):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except Exception as exc:
            errors.append(f"json:{exc}")
            continue

        name = data.get("name")
        arguments = data.get("arguments", {})
        if name is None and isinstance(data.get("function"), dict):
            name = data["function"].get("name")
            arguments = data["function"].get("arguments", arguments)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception as exc:
                errors.append(f"arguments_json:{exc}")
                arguments = {}
        if not isinstance(arguments, dict):
            errors.append("arguments_not_object")
            arguments = {}

        calls.append(
            {
                "name": str(name) if name is not None else "",
                "arguments": arguments,
                "raw": raw,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return calls, errors


def _parse_number_list(value, expected_len: int) -> list[float] | None:
    try:
        if isinstance(value, (list, tuple)):
            nums = [float(x) for x in value]
        elif isinstance(value, str):
            text = value.strip().strip("[]()")
            nums = [float(x.strip()) for x in text.split(",")]
        else:
            return None
    except Exception:
        return None
    if len(nums) != expected_len:
        return None
    return nums


def _bbox_is_valid(value) -> bool:
    nums = _parse_number_list(value, 4)
    if nums is None:
        return False
    xmin, ymin, xmax, ymax = nums
    return all(0 <= v <= 1000 for v in nums) and xmax > xmin and ymax > ymin


def _position_is_valid(value) -> bool:
    nums = _parse_number_list(value, 2)
    if nums is None:
        return False
    return all(0 <= v <= 1000 for v in nums)


def _normalise_arg_value(value):
    if isinstance(value, dict):
        return {k: _normalise_arg_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalise_arg_value(v) for v in value]
    return value


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    punct_chunks = max(0, len(text) - cjk) // 8
    return cjk + words + punct_chunks


def _score_format(solution_str: str, calls: list[dict], parse_errors: list[str]) -> tuple[float, dict]:
    tool_blocks = len(_TOOL_CALL_RE.findall(solution_str or ""))
    reason_blocks = _REASON_RE.findall(solution_str or "")

    has_valid_reason_tags = 1.0 if reason_blocks else 0.0
    if reason_blocks and len(reason_blocks) < max(1, tool_blocks):
        has_valid_reason_tags = 0.7

    valid_tool_call_json = 1.0
    if tool_blocks:
        valid_tool_call_json = max(0.0, len(calls) / tool_blocks)
    elif parse_errors:
        valid_tool_call_json = 0.0

    segments = re.split(r"<\|im_end\|>|<\|endoftext\|>", solution_str or "")
    if len(segments) > 1:
        max_calls_per_segment = max(seg.count("<tool_call>") for seg in segments)
        at_most_one_tool_per_turn = 1.0 if max_calls_per_segment <= 1 else 0.0
    else:
        at_most_one_tool_per_turn = 1.0

    terminate_count = (solution_str or "").count("<Terminate>")
    if terminate_count == 1:
        last_term = (solution_str or "").rfind("<Terminate>")
        valid_terminate_usage = 1.0 if "<tool_call>" not in solution_str[last_term:] else 0.3
    elif terminate_count == 0:
        valid_terminate_usage = 0.4
    else:
        valid_terminate_usage = 0.5

    no_extra_garbage_text = 0.0 if "```" in (solution_str or "") else 1.0
    if re.search(r"</?tool_calls?>", solution_str or "") and tool_blocks == 0:
        no_extra_garbage_text = 0.4

    score = (
        0.25 * has_valid_reason_tags
        + 0.30 * valid_tool_call_json
        + 0.20 * at_most_one_tool_per_turn
        + 0.15 * valid_terminate_usage
        + 0.10 * no_extra_garbage_text
    )
    details = {
        "tool_blocks": tool_blocks,
        "reason_blocks": len(reason_blocks),
        "parse_error_count": len(parse_errors),
        "terminate_count": terminate_count,
        "has_valid_reason_tags": has_valid_reason_tags,
        "valid_tool_call_json": valid_tool_call_json,
        "at_most_one_tool_per_turn": at_most_one_tool_per_turn,
        "valid_terminate_usage": valid_terminate_usage,
        "no_extra_garbage_text": no_extra_garbage_text,
    }
    return _clamp01(score), details


def _initial_image_ids(extra_info: dict) -> set[str]:
    ids = set()
    has_input = bool(extra_info.get("input_image_b64")) or bool(extra_info.get("image_paths"))
    if has_input:
        ids.update({"img_1", "base", "base_image"})
    image_paths = extra_info.get("image_paths")
    if isinstance(image_paths, (list, tuple)):
        ids.update(f"img_{idx + 1}" for idx in range(len(image_paths)))
    for key in ("imglist", "available_assets"):
        value = extra_info.get(key)
        if isinstance(value, str):
            candidates = _IMAGE_ID_TOKEN_RE.findall(value)
        elif isinstance(value, (list, tuple, set)):
            candidates = [str(v) for v in value if _IMAGE_ID_RE.match(str(v))]
        elif isinstance(value, dict):
            candidates = [str(k) for k in value.keys() if _IMAGE_ID_RE.match(str(k))]
        else:
            candidates = []
        # Only initial image aliases should be trusted at the start. Tool outputs
        # are added sequentially below after the corresponding tool call.
        ids.update(x for x in candidates if x.startswith("img_") or x in {"base", "base_image"})
    return ids


def _image_ref_score(ref: str | None, known_image_ids: set[str], has_input: bool) -> tuple[float, bool]:
    if not ref:
        return (0.8 if has_input else 0.0), False
    if ref in {"base", "base_image"} and has_input:
        return 1.0, False
    if ref in known_image_ids:
        return 1.0, False
    if _IMAGE_ID_RE.match(ref):
        return 0.0, True
    return 0.0, True


def _is_valid_overlayer_object_ref(content: str | None) -> bool:
    """Allow either an Extract output or a secondary input image like img_2.

    The standard object-overlay chain is Extract -> Overlayer(object), but some
    single-tool datasets provide the transparent RGBA object directly as a
    second input image. In that case the runtime image ID is img_2 (or img_3,
    etc.), which should not be treated as a critical chain error.
    """
    if not isinstance(content, str):
        return False
    if re.match(r"^Extract_\d+$", content):
        return True
    m = re.match(r"^img_(\d+)$", content)
    return bool(m and int(m.group(1)) >= 2)


def _score_action_process(
    calls: list[dict],
    expected_tools: list[str],
    extra_info: dict,
) -> tuple[float, dict]:
    if not calls:
        return 0.0, {
            "invalid_tool_count": 0,
            "invalid_image_ref_count": 0,
            "critical_chain_error_count": 0,
            "tool_count": 0,
            "executable_call_count": 0,
        }

    tool_rewards = _normalise_tool_rewards(extra_info.get("tool_rewards", []))
    has_input = bool(extra_info.get("input_image_b64")) or bool(extra_info.get("image_paths")) or "img_1" in _initial_image_ids(extra_info)
    known_image_ids = _initial_image_ids(extra_info)
    generated_counts = {tool: 0 for tool in VALID_TOOLS}
    called_so_far = []
    per_call_scores = []
    invalid_tool_count = 0
    invalid_image_ref_count = 0
    critical_chain_errors = []
    executable_call_count = 0

    for idx, call in enumerate(calls):
        name = _canonical_tool_name(call.get("name"))
        args = call.get("arguments") or {}

        tool_name_valid = 1.0 if name else 0.0
        if not name:
            invalid_tool_count += 1

        required = REQUIRED_ARGS.get(name or "", ())
        schema_valid = 1.0 if all(k in args and args.get(k) not in (None, "") for k in required) else 0.0

        refs_to_check = []
        for key in IMAGE_REF_KEYS:
            if key in args and isinstance(args.get(key), str):
                refs_to_check.append(args.get(key))
        overlay_type = args.get("overlay_type")
        if name == "Overlayer" and overlay_type == "object" and isinstance(args.get("content"), str):
            refs_to_check.append(args.get("content"))

        if refs_to_check:
            ref_scores = []
            for ref in refs_to_check:
                ref_score, invalid_ref = _image_ref_score(ref, known_image_ids, has_input)
                ref_scores.append(ref_score)
                if invalid_ref:
                    invalid_image_ref_count += 1
            image_id_valid = sum(ref_scores) / len(ref_scores)
        else:
            # Many tools default to the first input image if image is omitted.
            image_id_valid = 1.0 if name == "ImageGeneration" or has_input else 0.0

        tool_specific = 1.0
        if name in {"Crop", "SAM"}:
            bbox = args.get("bbox") if "bbox" in args else args.get("bounding_box")
            tool_specific = 1.0 if _bbox_is_valid(bbox) else 0.0
            if name == "SAM" and "Grounding" in expected_tools and "Grounding" not in called_so_far:
                tool_specific = min(tool_specific, 0.7)
        elif name == "Grounding":
            tool_specific = 1.0 if isinstance(args.get("reference_text"), str) and args.get("reference_text").strip() else 0.0
        elif name == "Rotate":
            try:
                tool_specific = 1.0 if int(args.get("angle")) in {45, 90, 180} else 0.0
            except Exception:
                tool_specific = 0.0
        elif name == "Extract":
            mask = args.get("mask") or args.get("mask_ref")
            tool_specific = 1.0 if isinstance(mask, str) and re.match(r"^SAM_\d+$", mask) else 0.0
            if tool_specific == 0.0:
                critical_chain_errors.append("extract_mask_not_sam")
        elif name == "Overlayer":
            overlay_type = args.get("overlay_type")
            if overlay_type not in {"text", "object"}:
                tool_specific = 0.0
            elif overlay_type == "object":
                content = args.get("content")
                # Object overlays require a transparent RGBA object input.
                # Accept either a prior Extract output or a secondary input
                # image such as img_2 when the dataset directly provides the
                # transparent object as an input image.
                tool_specific = 1.0 if _is_valid_overlayer_object_ref(content) else 0.0
                if tool_specific == 0.0:
                    critical_chain_errors.append("overlayer_object_not_extract_rgba")
            if "position" in args and not _position_is_valid(args.get("position")):
                tool_specific = 0.0

        if idx < len(tool_rewards):
            execution_success = 1.0 if _is_numeric_tool_reward(tool_rewards[idx]) and tool_rewards[idx] >= 0 else 0.0
        else:
            execution_success = 1.0

        per_call = (
            0.25 * tool_name_valid
            + 0.25 * schema_valid
            + 0.20 * image_id_valid
            + 0.20 * tool_specific
            + 0.10 * execution_success
        )
        per_call_scores.append(_clamp01(per_call))

        # "Executable" means the parsed tool call is valid enough to be run:
        # the tool name exists, required arguments are present, image refs are
        # valid, and tool-specific constraints are satisfied. We do not require
        # the runtime execution itself to succeed here.
        if (
            tool_name_valid >= 1.0
            and schema_valid >= 1.0
            and image_id_valid >= 1.0
            and tool_specific >= 1.0
        ):
            executable_call_count += 1

        if name:
            called_so_far.append(name)
            if name in IMAGE_OUTPUT_TOOLS and execution_success > 0:
                output_id = f"{name}_{generated_counts[name]}"
                known_image_ids.add(output_id)
                generated_counts[name] += 1

    score = sum(per_call_scores) / len(per_call_scores)
    details = {
        "tool_count": len(calls),
        "invalid_tool_count": invalid_tool_count,
        "invalid_image_ref_count": invalid_image_ref_count,
        "critical_chain_error_count": len(critical_chain_errors),
        "critical_chain_errors": "|".join(critical_chain_errors) if critical_chain_errors else "none",
        "executable_call_count": executable_call_count,
        "per_call_avg": score,
    }
    return _clamp01(score), details


def _compute_repeat_penalty(calls: list[dict]) -> tuple[float, dict]:
    exact_repeats = 0
    same_tool_repeats = 0
    prev = None
    for call in calls:
        cur = (
            _canonical_tool_name(call.get("name")) or call.get("name"),
            json.dumps(_normalise_arg_value(call.get("arguments") or {}), sort_keys=True, ensure_ascii=False),
        )
        if prev:
            if cur == prev:
                exact_repeats += 1
            elif cur[0] == prev[0]:
                same_tool_repeats += 1
        prev = cur
    penalty = min(0.04 * exact_repeats + 0.02 * same_tool_repeats, 0.12)
    return penalty, {"exact_repeat_count": exact_repeats, "same_tool_repeat_count": same_tool_repeats}


def _compute_length_penalty(solution_str: str) -> tuple[float, dict]:
    reason_blocks = _REASON_RE.findall(solution_str or "")
    if not reason_blocks:
        return 0.0, {"overlong_turn_count": 0, "max_reason_tokens": 0}
    max_tokens = 0
    penalty = 0.0
    overlong_count = 0
    for block in reason_blocks:
        tokens = _estimate_tokens(block)
        max_tokens = max(max_tokens, tokens)
        excess = max(0, tokens - TURN_TOKEN_LIMIT)
        if excess:
            overlong_count += 1
            penalty += min((excess / TURN_TOKEN_LIMIT) * 0.03, 0.05)
    return min(penalty, 0.10), {"overlong_turn_count": overlong_count, "max_reason_tokens": max_tokens}


def _compute_tool_coverage_reward(expected_tools: list[str], coverage_details: dict) -> float:
    if not expected_tools:
        return 0.0
    return TOOL_COVERAGE_REWARD_MAX * _clamp01(coverage_details.get("coverage_ratio", 0.0))


def _compute_interaction_reward(
    calls: list[dict],
    expected_tool_path: list[str],
    action_details: dict,
) -> tuple[float, dict]:
    expected_turns = len(expected_tool_path)
    observed_turns = len(calls)
    executable_turns = int(action_details.get("executable_call_count", 0))

    if expected_turns <= 1:
        return 0.0, {
            "expected_interaction_turns": expected_turns,
            "observed_interaction_turns": observed_turns,
            "executable_tool_turns": executable_turns,
            "interaction_depth_ratio": 0.0,
            "multi_turn_activation": 0.0,
            "interaction_reward_ratio": 0.0,
        }

    matched_turns = min(executable_turns, expected_turns)
    interaction_depth_ratio = matched_turns / max(expected_turns, 1)
    # Strongly reward crossing into a true multi-turn trajectory (2+ executable
    # tool steps), while still scaling with full expected depth afterwards.
    first_turn_ratio = 1.0 if executable_turns >= 1 else 0.0
    multi_turn_target = min(expected_turns, 2)
    multi_turn_activation = 1.0 if executable_turns >= multi_turn_target else 0.0
    reward_ratio = (
        0.15 * first_turn_ratio
        + 0.35 * multi_turn_activation
        + 0.50 * interaction_depth_ratio
    )
    reward = INTERACTION_REWARD_MAX * _clamp01(reward_ratio)
    return reward, {
        "expected_interaction_turns": expected_turns,
        "observed_interaction_turns": observed_turns,
        "executable_tool_turns": executable_turns,
        "interaction_depth_ratio": interaction_depth_ratio,
        "multi_turn_activation": multi_turn_activation,
        "interaction_reward_ratio": reward_ratio,
    }


def _compute_missing_key_tool_penalty(
    expected_tools: list[str],
    expected_tool_path: list[str],
    coverage_details: dict,
) -> float:
    if not expected_tools:
        return 0.0
    missing_count = len(coverage_details.get("missing_expected_tools", ()))
    missing_ratio = missing_count / max(len(expected_tools), 1)
    expected_path_len = len(expected_tool_path)

    if expected_path_len <= 2:
        penalty_scale = 1.0
    elif expected_path_len <= 4:
        penalty_scale = 0.85
    else:
        # Long chains naturally miss more intermediate steps during exploration.
        # Keep the penalty, but relax it so these tasks are not dominated by a
        # single missing-tool term before the policy learns deeper trajectories.
        penalty_scale = 0.65

    return MISSING_KEY_TOOL_PENALTY_MAX * penalty_scale * _clamp01(missing_ratio)


def _compute_tool_cost_penalty(calls: list[dict], expected_tool_path: list[str]) -> float:
    expected_tool_budget = len(expected_tool_path)
    if expected_tool_budget <= 0:
        return 0.0

    if expected_tool_budget <= 2:
        extra_grace = 2
        per_extra_penalty = 0.006
        cap = 0.03
    elif expected_tool_budget <= 4:
        extra_grace = 3
        per_extra_penalty = 0.0045
        cap = 0.025
    else:
        # Long tool chains naturally need more slack; avoid teaching the model
        # to collapse to unnaturally short trajectories just to save tool cost.
        extra_grace = max(4, expected_tool_budget // 2)
        per_extra_penalty = 0.002
        cap = 0.015

    extra_count = max(0, len(calls) - expected_tool_budget - extra_grace)
    return min(per_extra_penalty * extra_count, cap)


def _normalise_tool_rewards(tool_rewards) -> list:
    if tool_rewards is None:
        return []
    if hasattr(tool_rewards, "tolist"):
        tool_rewards = tool_rewards.tolist()
    if isinstance(tool_rewards, list):
        return tool_rewards
    if isinstance(tool_rewards, tuple):
        return list(tool_rewards)
    try:
        return list(tool_rewards)
    except TypeError:
        return [tool_rewards]


def _is_numeric_tool_reward(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _compute_fatal_error(
    solution_str: str,
    calls: list[dict],
    parse_errors: list[str],
    expected_tools: list[str],
    action_details: dict,
    final_image_b64: str | None,
    tool_rewards: list,
) -> tuple[float, str]:
    cap = 1.0
    reasons = []
    task_requires_tool = bool(expected_tools)
    task_requires_image = bool(set(expected_tools) & IMAGE_TASK_TOOLS)

    def apply(value: float, reason: str):
        nonlocal cap
        cap = min(cap, value)
        reasons.append(reason)

    if task_requires_tool and action_details.get("executable_call_count", 0) <= 0:
        apply(0.0, "no_executable_tool_call")
    if task_requires_image and not final_image_b64:
        apply(0.0, "missing_final_image")

    return cap, "|".join(reasons) if reasons else "none"


def _sample_identifiers(extra_info: dict, kwargs: dict, task_text: str) -> tuple[str, str]:
    step = _first_non_empty(
        kwargs.get("global_step"),
        kwargs.get("step"),
        extra_info.get("global_step"),
        extra_info.get("step"),
    )
    task_id = _first_non_empty(
        extra_info.get("task_id"),
        extra_info.get("sample_id"),
        extra_info.get("id"),
        extra_info.get("uid"),
        _task_hash(task_text),
    )
    return str(step), str(task_id)


def _invalid_tool_names(calls: list[dict]) -> list[str]:
    names = []
    for call in calls:
        name = str(call.get("name") or "")
        if name and not _canonical_tool_name(name):
            names.append(name)
    return names


def _log_reward_sample(
    *,
    step: str,
    task_id: str,
    score: float,
    raw_score: float,
    normalized_score: float,
    fatal_error: float,
    fatal_error_reasons: str,
    image_prompt_judge_score: float,
    aesthetic_judge_score: float,
    trajectory_judge_score: float,
    image_prompt_api_failed: bool,
    aesthetic_api_failed: bool,
    trajectory_api_failed: bool,
    format_score: float,
    action_process_score: float,
    outcome_reward: float,
    process_reward: float,
    tool_coverage_reward: float,
    interaction_reward: float,
    efficiency_penalty: float,
    error_penalty: float,
    repeat_penalty: float,
    length_penalty: float,
    tool_cost_penalty: float,
    missing_key_tool_penalty: float,
    error_count: int,
    tool_rewards: list,
    final_image_b64: str | None,
    input_image_b64: str | None,
    calls: list[dict],
    action_details: dict,
    coverage_details: dict,
    interaction_details: dict,
):
    if not _log_enabled("sample"):
        return
    call_names = [str(call.get("name") or "") for call in calls if call.get("name")]
    call_names_short = "|".join(call_names[:8]) if call_names else "none"
    if len(call_names) > 8:
        call_names_short += f"|+{len(call_names) - 8}"
    invalid_names = _invalid_tool_names(calls)
    invalid_names_short = "|".join(invalid_names[:5]) if invalid_names else "none"
    if len(invalid_names) > 5:
        invalid_names_short += f"|+{len(invalid_names) - 5}"
    print(
        f"[REWARD_SAMPLE] pid={os.getpid()} step={step} task={task_id} "
        f"score={score:.4f} raw={raw_score:.4f} norm={normalized_score:.4f} "
        f"fatal={fatal_error:.2f} reasons={fatal_error_reasons} "
        f"judge={image_prompt_judge_score:.3f}/{aesthetic_judge_score:.3f}/{trajectory_judge_score:.3f} "
        f"judge_fail=img:{int(image_prompt_api_failed)},aes:{int(aesthetic_api_failed)},traj:{int(trajectory_api_failed)} "
        f"fmt={format_score:.3f} action={action_process_score:.3f} "
        f"outcome={outcome_reward:.3f} process={process_reward:.3f} "
        f"coverage={coverage_details.get('coverage_ratio', 0.0):.2f} cov_reward={tool_coverage_reward:.2f} "
        f"interaction=reward:{interaction_reward:.2f},ratio:{interaction_details.get('interaction_depth_ratio', 0.0):.2f} "
        f"penalty=eff:{efficiency_penalty:.2f},err:{error_penalty:.2f},rep:{repeat_penalty:.2f},len:{length_penalty:.2f},tool:{tool_cost_penalty:.2f},miss:{missing_key_tool_penalty:.2f} "
        f"image=input:{int(input_image_b64 is not None)},final:{int(final_image_b64 is not None)} "
        f"tools=calls:{len(calls)},errors:{error_count}/{len(tool_rewards)},invalid:{action_details.get('invalid_tool_count', 0)} "
        f"invalid_names={invalid_names_short} names={call_names_short}",
        flush=True,
    )


def _update_reward_window_summary(
    *,
    score: float,
    fatal_error: float,
    fatal_error_reasons: str,
    error_count: int,
    image_prompt_api_failed: bool,
    aesthetic_api_failed: bool,
    trajectory_api_failed: bool,
    invalid_tool_names: list[str],
):
    if not _log_enabled("summary") or REWARD_SUMMARY_EVERY <= 0:
        return

    stats = _REWARD_WINDOW_STATS
    stats["n"] += 1
    stats["scores"].append(float(score))
    if score <= 0.0:
        stats["zero"] += 1
    if fatal_error < 1.0:
        stats["fatal"] += 1
    stats["tool_errors"] += int(error_count)
    if image_prompt_api_failed:
        stats["judge_fail"]["img"] += 1
    if aesthetic_api_failed:
        stats["judge_fail"]["aes"] += 1
    if trajectory_api_failed:
        stats["judge_fail"]["traj"] += 1
    for reason in (fatal_error_reasons or "").split("|"):
        if reason and reason != "none":
            stats["fatal_reasons"][reason] += 1
    for name in invalid_tool_names:
        stats["invalid_tools"][name] += 1

    if stats["n"] < REWARD_SUMMARY_EVERY:
        return

    scores = stats["scores"]
    mean = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    print(
        f"[REWARD_WINDOW_SUMMARY] pid={os.getpid()} n={stats['n']} "
        f"mean={mean:.4f} min={min_score:.4f} max={max_score:.4f} "
        f"zero={stats['zero']} fatal={stats['fatal']} tool_errors={stats['tool_errors']} "
        f"judge_fail={_format_counter(stats['judge_fail'])} "
        f"fatal_reasons={_format_counter(stats['fatal_reasons'])} "
        f"invalid_tools={_format_counter(stats['invalid_tools'])}",
        flush=True,
    )
    stats["n"] = 0
    stats["scores"].clear()
    stats["zero"] = 0
    stats["fatal"] = 0
    stats["tool_errors"] = 0
    stats["judge_fail"].clear()
    stats["fatal_reasons"].clear()
    stats["invalid_tools"].clear()


def _write_reward_debug(payload: dict):
    if _log_enabled("debug"):
        _write_debug_jsonl("reward_debug", payload)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute hybrid trajectory reward.

    Args:
        data_source: dataset identifier (unused, kept for API compatibility)
        solution_str: full generated response (all turns concatenated)
        ground_truth: task description or expected tool path
        extra_info: optional dict; recognised keys:
            - ``final_image_b64``: base64 JPEG of the last image produced
            - ``input_image_b64``: base64 JPEG of the input image
            - ``tool_rewards``: step-level tool execution rewards
            - ``tools``/``tool_path``: expected tool set/path

    Returns:
        dict with normalized ``score`` in [0.0, 1.0] and diagnostic keys.
    """
    if not solution_str or not isinstance(solution_str, str):
        return {
            "score": 0.0,
            "judge_score": 0.0,
            "image_prompt_judge_score": 0.0,
            "aesthetic_judge_score": 0.0,
            "trajectory_judge_score": 0.0,
            "format_score": 0.0,
            "action_process_score": 0.0,
            "outcome_reward": 0.0,
            "process_reward": 0.0,
            "tool_coverage_reward": 0.0,
            "interaction_reward": 0.0,
            "interaction_depth_ratio": 0.0,
            "expected_interaction_turns": 0,
            "observed_interaction_turns": 0,
            "executable_tool_turns": 0,
            "api_failed": False,
            "error_penalty": 0.0,
            "error_count": 0,
        }

    extra_info = extra_info or {}

    expected_tool_path = _extract_expected_tool_path(ground_truth, extra_info)
    expected_tools = _unique_preserve_order(expected_tool_path)
    task_text = _extract_task_text(ground_truth, extra_info)

    final_image_b64 = extra_info.get("final_image_b64")
    input_image_b64 = extra_info.get("input_image_b64")
    tool_rewards = _normalise_tool_rewards(extra_info.get("tool_rewards", []))
    error_count = sum(1 for r in tool_rewards if _is_numeric_tool_reward(r) and r < 0)

    calls, parse_errors = _parse_tool_calls(solution_str)
    format_score, format_details = _score_format(solution_str, calls, parse_errors)
    action_process_score, action_details = _score_action_process(calls, expected_tools, extra_info)
    coverage_details = _summarise_expected_tool_coverage(calls, expected_tools)

    repeat_penalty, repeat_details = _compute_repeat_penalty(calls)
    length_penalty, length_details = _compute_length_penalty(solution_str)
    tool_coverage_reward = _compute_tool_coverage_reward(expected_tools, coverage_details)
    interaction_reward, interaction_details = _compute_interaction_reward(calls, expected_tool_path, action_details)
    missing_key_tool_penalty = _compute_missing_key_tool_penalty(
        expected_tools,
        expected_tool_path,
        coverage_details,
    )
    tool_cost_penalty = _compute_tool_cost_penalty(calls, expected_tool_path)

    image_prompt_judge_score = _call_image_prompt_judge(task_text, solution_str, final_image_b64, input_image_b64)
    image_prompt_api_failed = image_prompt_judge_score is None
    if image_prompt_api_failed:
        image_prompt_judge_score = FALLBACK_SCORE

    aesthetic_judge_score = _call_aesthetic_judge(task_text, final_image_b64, input_image_b64)
    aesthetic_api_failed = aesthetic_judge_score is None
    if aesthetic_api_failed:
        aesthetic_judge_score = FALLBACK_SCORE

    trajectory_judge_score = _call_trajectory_judge(
        task_text,
        solution_str,
        expected_tools,
        calls,
        error_count,
    )
    trajectory_api_failed = trajectory_judge_score is None
    if trajectory_api_failed:
        trajectory_judge_score = FALLBACK_SCORE

    judge_score = (
        WEIGHT_IMAGE_PROMPT_JUDGE * image_prompt_judge_score
        + WEIGHT_AESTHETIC_JUDGE * aesthetic_judge_score
        + WEIGHT_TRAJECTORY_JUDGE * trajectory_judge_score
    ) / max(WEIGHT_IMAGE_PROMPT_JUDGE + WEIGHT_AESTHETIC_JUDGE + WEIGHT_TRAJECTORY_JUDGE, 1e-6)

    outcome_positive = (
        WEIGHT_IMAGE_PROMPT_JUDGE * image_prompt_judge_score
        + WEIGHT_AESTHETIC_JUDGE * aesthetic_judge_score
    )
    if OUTCOME_REWARD_COMPONENT_MAX > 1e-6:
        outcome_reward = OUTCOME_REWARD_BUDGET * _clamp01(outcome_positive / OUTCOME_REWARD_COMPONENT_MAX)
    else:
        outcome_reward = 0.0

    process_positive = (
        WEIGHT_TRAJECTORY_JUDGE * trajectory_judge_score
        + WEIGHT_FORMAT * format_score
        + WEIGHT_ACTION_PROCESS * action_process_score
        + tool_coverage_reward
        + interaction_reward
    )
    if PROCESS_REWARD_COMPONENT_MAX > 1e-6:
        process_reward = PROCESS_REWARD_BUDGET * _clamp01(process_positive / PROCESS_REWARD_COMPONENT_MAX)
    else:
        process_reward = 0.0

    base_score = outcome_reward + process_reward

    # Tool reward values: 0.0 = success, -0.05 = tool internal error, -0.1 = call exception.
    error_penalty = 0.0
    if error_count > 0:
        error_penalty = min(error_count * 0.1, 0.5)  # cap at 0.5

    efficiency_penalty = error_penalty + repeat_penalty + length_penalty + tool_cost_penalty + missing_key_tool_penalty
    raw_score = base_score - efficiency_penalty
    normalized_score = _normalize_total_reward(raw_score)
    fatal_error, fatal_error_reasons = _compute_fatal_error(
        solution_str=solution_str,
        calls=calls,
        parse_errors=parse_errors,
        expected_tools=expected_tools,
        action_details=action_details,
        final_image_b64=final_image_b64,
        tool_rewards=tool_rewards,
    )
    score = fatal_error * normalized_score

    step, task_id = _sample_identifiers(extra_info, kwargs, task_text)
    invalid_names = _invalid_tool_names(calls)
    _log_reward_sample(
        step=step,
        task_id=task_id,
        score=score,
        raw_score=raw_score,
        normalized_score=normalized_score,
        fatal_error=fatal_error,
        fatal_error_reasons=fatal_error_reasons,
        image_prompt_judge_score=image_prompt_judge_score,
        aesthetic_judge_score=aesthetic_judge_score,
        trajectory_judge_score=trajectory_judge_score,
        image_prompt_api_failed=image_prompt_api_failed,
        aesthetic_api_failed=aesthetic_api_failed,
        trajectory_api_failed=trajectory_api_failed,
        format_score=format_score,
        action_process_score=action_process_score,
        outcome_reward=outcome_reward,
        process_reward=process_reward,
        tool_coverage_reward=tool_coverage_reward,
        interaction_reward=interaction_reward,
        efficiency_penalty=efficiency_penalty,
        error_penalty=error_penalty,
        repeat_penalty=repeat_penalty,
        length_penalty=length_penalty,
        tool_cost_penalty=tool_cost_penalty,
        missing_key_tool_penalty=missing_key_tool_penalty,
        error_count=error_count,
        tool_rewards=tool_rewards,
        final_image_b64=final_image_b64,
        input_image_b64=input_image_b64,
        calls=calls,
        action_details=action_details,
        coverage_details=coverage_details,
        interaction_details=interaction_details,
    )
    _update_reward_window_summary(
        score=score,
        fatal_error=fatal_error,
        fatal_error_reasons=fatal_error_reasons,
        error_count=error_count,
        image_prompt_api_failed=image_prompt_api_failed,
        aesthetic_api_failed=aesthetic_api_failed,
        trajectory_api_failed=trajectory_api_failed,
        invalid_tool_names=invalid_names,
    )
    _write_reward_debug(
        {
            "step": step,
            "task_id": task_id,
            "data_source": data_source,
            "task": task_text,
            "score": score,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "uncapped_score": raw_score,
            "fatal_error": fatal_error,
            "fatal_error_reasons": fatal_error_reasons,
            "judge": {
                "image_prompt": image_prompt_judge_score,
                "aesthetic": aesthetic_judge_score,
                "trajectory": trajectory_judge_score,
                "api_failed": image_prompt_api_failed or aesthetic_api_failed or trajectory_api_failed,
            },
            "format_score": format_score,
            "action_process_score": action_process_score,
            "outcome_reward": outcome_reward,
            "process_reward": process_reward,
            "tool_coverage_reward": tool_coverage_reward,
            "interaction_reward": interaction_reward,
            "efficiency_penalty": efficiency_penalty,
            "penalties": {
                "error": error_penalty,
                "repeat": repeat_penalty,
                "length": length_penalty,
                "tool_cost": tool_cost_penalty,
                "missing_key_tool": missing_key_tool_penalty,
            },
            "tool_rewards": tool_rewards,
            "expected_tools": expected_tools,
            "expected_tool_path": expected_tool_path,
            "calls": [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls],
            "format_details": format_details,
            "action_details": action_details,
            "coverage_details": coverage_details,
            "interaction_details": interaction_details,
            "repeat_details": repeat_details,
            "length_details": length_details,
        }
    )

    return {
        "score": score,
        "judge_score": judge_score,
        "image_prompt_judge_score": image_prompt_judge_score,
        "aesthetic_judge_score": aesthetic_judge_score,
        "trajectory_judge_score": trajectory_judge_score,
        "format_score": format_score,
        "action_process_score": action_process_score,
        "outcome_reward": outcome_reward,
        "process_reward": process_reward,
        "tool_coverage_reward": tool_coverage_reward,
        "interaction_reward": interaction_reward,
        "efficiency_penalty": efficiency_penalty,
        "base_score": base_score,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "uncapped_score": raw_score,
        "fatal_error": fatal_error,
        "repeat_penalty": repeat_penalty,
        "length_penalty": length_penalty,
        "tool_cost_penalty": tool_cost_penalty,
        "missing_key_tool_penalty": missing_key_tool_penalty,
        "error_penalty": error_penalty,
        "error_count": error_count,
        "api_failed": image_prompt_api_failed or aesthetic_api_failed or trajectory_api_failed,
        "image_prompt_api_failed": image_prompt_api_failed,
        "aesthetic_api_failed": aesthetic_api_failed,
        "trajectory_api_failed": trajectory_api_failed,
        "tool_call_count": len(calls),
        "expected_tool_count": coverage_details.get("expected_tool_count", 0),
        "expected_tool_path_length": len(expected_tool_path),
        "tool_coverage_ratio": coverage_details.get("coverage_ratio", 0.0),
        "interaction_depth_ratio": interaction_details.get("interaction_depth_ratio", 0.0),
        "expected_interaction_turns": interaction_details.get("expected_interaction_turns", 0),
        "observed_interaction_turns": interaction_details.get("observed_interaction_turns", len(calls)),
        "executable_tool_turns": interaction_details.get("executable_tool_turns", 0),
        "missing_expected_tool_count": len(coverage_details.get("missing_expected_tools", ())),
        "parse_error_count": len(parse_errors),
        "invalid_tool_count": action_details.get("invalid_tool_count", 0),
        "invalid_image_ref_count": action_details.get("invalid_image_ref_count", 0),
        "critical_chain_error_count": action_details.get("critical_chain_error_count", 0),
        "fatal_error_reasons": fatal_error_reasons,
        "format_details": json.dumps(format_details, ensure_ascii=False),
        "action_details": json.dumps(action_details, ensure_ascii=False),
        "coverage_details": json.dumps(coverage_details, ensure_ascii=False),
        "interaction_details": json.dumps(interaction_details, ensure_ascii=False),
        "repeat_details": json.dumps(repeat_details, ensure_ascii=False),
        "length_details": json.dumps(length_details, ensure_ascii=False),
    }
