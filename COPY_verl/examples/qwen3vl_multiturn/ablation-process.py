#!/usr/bin/env python3
"""
Process-ablation reward for multi-turn visual tool-use RL training.

This reward keeps the outcome judges, efficiency penalties, and fatal error
gate from ``multiturn_reward.py`` unchanged, while removing the process layer
from the final reward:
  - trajectory_judge
  - format_score
  - action_process_score
  - tool_coverage_reward
  - interaction_reward

For monitoring consistency, those process-related signals are still computed
and returned in the diagnostics so validation curves remain directly
comparable with the original ``multiturn_reward.py`` setup.

The final reward is still normalized to [0, 1]. Since the process layer is
removed from optimization, the full positive budget is reassigned to the
outcome layer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("multiturn_reward.py")
_SPEC = importlib.util.spec_from_file_location("multiturn_reward_base_ablation_process", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Failed to load base reward module from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


WEIGHT_IMAGE_PROMPT_JUDGE = _BASE.WEIGHT_IMAGE_PROMPT_JUDGE
WEIGHT_AESTHETIC_JUDGE = _BASE.WEIGHT_AESTHETIC_JUDGE
WEIGHT_TRAJECTORY_JUDGE = _BASE.WEIGHT_TRAJECTORY_JUDGE
WEIGHT_FORMAT = _BASE.WEIGHT_FORMAT
WEIGHT_ACTION_PROCESS = _BASE.WEIGHT_ACTION_PROCESS
TOOL_COVERAGE_REWARD_MAX = _BASE.TOOL_COVERAGE_REWARD_MAX
INTERACTION_REWARD_MAX = _BASE.INTERACTION_REWARD_MAX
TOTAL_REWARD_POSITIVE_MAX = _BASE.TOTAL_REWARD_POSITIVE_MAX
OUTCOME_REWARD_COMPONENT_MAX = WEIGHT_IMAGE_PROMPT_JUDGE + WEIGHT_AESTHETIC_JUDGE
PROCESS_REWARD_COMPONENT_MAX = _BASE.PROCESS_REWARD_COMPONENT_MAX
OUTCOME_REWARD_BUDGET = min(TOTAL_REWARD_POSITIVE_MAX, TOTAL_REWARD_POSITIVE_MAX)
PROCESS_REWARD_BUDGET = 0.0
FALLBACK_SCORE = _BASE.FALLBACK_SCORE


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute ablated reward while preserving process diagnostics."""
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

    expected_tool_path = _BASE._extract_expected_tool_path(ground_truth, extra_info)
    expected_tools = _BASE._unique_preserve_order(expected_tool_path)
    task_text = _BASE._extract_task_text(ground_truth, extra_info)

    final_image_b64 = extra_info.get("final_image_b64")
    input_image_b64 = extra_info.get("input_image_b64")
    tool_rewards = _BASE._normalise_tool_rewards(extra_info.get("tool_rewards", []))
    error_count = sum(1 for r in tool_rewards if _BASE._is_numeric_tool_reward(r) and r < 0)

    calls, parse_errors = _BASE._parse_tool_calls(solution_str)
    format_score, format_details = _BASE._score_format(solution_str, calls, parse_errors)
    action_process_score, action_details = _BASE._score_action_process(calls, expected_tools, extra_info)
    coverage_details = _BASE._summarise_expected_tool_coverage(calls, expected_tools)

    repeat_penalty, repeat_details = _BASE._compute_repeat_penalty(calls)
    length_penalty, length_details = _BASE._compute_length_penalty(solution_str)
    tool_coverage_reward = _BASE._compute_tool_coverage_reward(expected_tools, coverage_details)
    interaction_reward, interaction_details = _BASE._compute_interaction_reward(
        calls,
        expected_tool_path,
        action_details,
    )
    missing_key_tool_penalty = _BASE._compute_missing_key_tool_penalty(
        expected_tools,
        expected_tool_path,
        coverage_details,
    )
    tool_cost_penalty = _BASE._compute_tool_cost_penalty(calls, expected_tool_path)

    image_prompt_judge_score = _BASE._call_image_prompt_judge(task_text, solution_str, final_image_b64, input_image_b64)
    image_prompt_api_failed = image_prompt_judge_score is None
    if image_prompt_api_failed:
        image_prompt_judge_score = FALLBACK_SCORE

    aesthetic_judge_score = _BASE._call_aesthetic_judge(task_text, final_image_b64, input_image_b64)
    aesthetic_api_failed = aesthetic_judge_score is None
    if aesthetic_api_failed:
        aesthetic_judge_score = FALLBACK_SCORE

    trajectory_judge_score = _BASE._call_trajectory_judge(
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
        outcome_reward = OUTCOME_REWARD_BUDGET * _BASE._clamp01(outcome_positive / OUTCOME_REWARD_COMPONENT_MAX)
    else:
        outcome_reward = 0.0

    process_reward = 0.0
    base_score = outcome_reward + process_reward

    error_penalty = 0.0
    if error_count > 0:
        error_penalty = min(error_count * 0.1, 0.5)

    efficiency_penalty = error_penalty + repeat_penalty + length_penalty + tool_cost_penalty + missing_key_tool_penalty
    raw_score = base_score - efficiency_penalty
    normalized_score = _BASE._normalize_total_reward(raw_score)
    fatal_error, fatal_error_reasons = _BASE._compute_fatal_error(
        solution_str=solution_str,
        calls=calls,
        parse_errors=parse_errors,
        expected_tools=expected_tools,
        action_details=action_details,
        final_image_b64=final_image_b64,
        tool_rewards=tool_rewards,
    )
    score = fatal_error * normalized_score

    step, task_id = _BASE._sample_identifiers(extra_info, kwargs, task_text)
    invalid_names = _BASE._invalid_tool_names(calls)
    _BASE._log_reward_sample(
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
    _BASE._update_reward_window_summary(
        score=score,
        fatal_error=fatal_error,
        fatal_error_reasons=fatal_error_reasons,
        error_count=error_count,
        image_prompt_api_failed=image_prompt_api_failed,
        aesthetic_api_failed=aesthetic_api_failed,
        trajectory_api_failed=trajectory_api_failed,
        invalid_tool_names=invalid_names,
    )
    _BASE._write_reward_debug(
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
