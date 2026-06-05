# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import copy
import io
import json
import logging
import os
import time

import torch
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_LOG_TOOL_DISPATCH = _env_flag("VERL_LOG_TOOL_DISPATCH", False)


def _prepare_jpeg_image(img):
    """Return an RGB image that Pillow can encode as JPEG."""
    mode = getattr(img, "mode", None)
    if mode == "RGB":
        return img

    has_alpha = mode in {"RGBA", "LA"} or (
        mode == "P" and "transparency" in getattr(img, "info", {})
    )
    if has_alpha:
        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img.getchannel("A"))
        return background

    return img.convert("RGB")


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Shared storage for tool outputs (images, etc.)
        # Format: {tool_name}_{index} -> image, e.g., "Extract_0" -> PIL.Image
        self.shared_tool_outputs: dict[str, Any] = {}


def _init_trace_logger():
    """Initialize a dedicated file logger for trajectory tracing."""
    trace_log_dir = os.environ.get("VERL_TRACE_LOG_DIR", "/data/zhuhairui/verl/examples/qwen3vl_multiturn/log")
    os.makedirs(trace_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:19]  # 毫秒精度：YYYYmmdd-HHMMSS_fff
    trace_log_path = os.path.join(trace_log_dir, f"trace_{timestamp}.log")

    trace_logger = logging.getLogger("verl.trace")
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False  # Don't propagate to root logger / stdout
    if not trace_logger.handlers:
        fh = logging.FileHandler(trace_log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        trace_logger.addHandler(fh)
    return trace_logger, trace_log_path


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    _trace_request_id: Optional[str] = None  # Only trace one trajectory at a time
    _trace_logger: Optional[logging.Logger] = None
    _trace_log_path: Optional[str] = None

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"[INIT] Initialized {len(cls.tools)} tools: {list(cls.tools.keys())}", flush=True)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Extract trace-save params (set by trainer at checkpoint steps)
        _save_trace = kwargs.pop("_save_trace", False)
        _trace_save_dir = kwargs.pop("_trace_save_dir", "")
        _sample_index = kwargs.pop("_sample_index", 0)
        _rollout_n = kwargs.pop("_rollout_n", 0)

        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Rollout progress: log trajectory start
        _start_time = datetime.now()
        _user_prompt = ""
        for _msg in messages:
            if _msg.get("role") == "user":
                _content = _msg.get("content", "")
                if isinstance(_content, list):
                    _user_prompt = " ".join(item.get("text", "") for item in _content if item.get("type") == "text")
                elif isinstance(_content, str):
                    _user_prompt = _content
                break
        print(f"[ROLLOUT] Start  req={request_id[:8]} prompt={_user_prompt[:80]!r}", flush=True)

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Select the first trajectory for tracing (write to dedicated log file)
        if ToolAgentLoop._trace_request_id is None:
            if ToolAgentLoop._trace_logger is None:
                ToolAgentLoop._trace_logger, ToolAgentLoop._trace_log_path = _init_trace_logger()
                print(f"[TRACE] Trace log file: {ToolAgentLoop._trace_log_path}")
            tl = ToolAgentLoop._trace_logger
            ToolAgentLoop._trace_request_id = request_id
            tl.info('#' * 80)
            tl.info(f"Tracing trajectory: {request_id}")
            tl.info(f"Initial prompt messages: {len(messages)} messages")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str):
                    preview = content[:200] + '...' if len(content) > 200 else content
                else:
                    preview = str(content)[:200] + '...'
                tl.info(f"  msg[{i}] role={role}: {preview}")
            tl.info('#' * 80)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # Register input images in shared_tool_outputs as img_1, img_2, ...
        # so the model can reference them by the IDs shown in the system prompt and tool descriptions.
        if image_data:
            img_list = image_data if isinstance(image_data, list) else [image_data]
            for i, img in enumerate(img_list):
                agent_data.shared_tool_outputs[f"img_{i + 1}"] = img
            logger.info(f"[DEBUG] request={request_id[:8]} registered {len(img_list)} input image(s) as {list(agent_data.shared_tool_outputs.keys())}")
        else:
            logger.info(f"[DEBUG] request={request_id[:8]} NO input images (image_data={type(image_data).__name__}:{image_data is None}), shared_tool_outputs is empty")

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Trace: log final summary to file
        if agent_data.request_id == ToolAgentLoop._trace_request_id:
            tl = ToolAgentLoop._trace_logger
            tl.info('#' * 80)
            tl.info(f"Trajectory completed: {agent_data.request_id}")
            tl.info(f"Assistant turns: {agent_data.assistant_turns}")
            tl.info(f"User/tool turns: {agent_data.user_turns}")
            tl.info(f"Total prompt tokens: {len(agent_data.prompt_ids)}")
            tl.info(f"Total response tokens: {len(agent_data.response_mask)}")
            tl.info(f"Tool rewards: {agent_data.tool_rewards}")
            tl.info(f"Turn scores: {agent_data.turn_scores}")
            tl.info('#' * 80)
            # Reset so next batch can trace a new trajectory
            ToolAgentLoop._trace_request_id = None

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        # Pass through the original image_grid_thw from the last processor call
        _saved_grid = getattr(agent_data, "_last_image_grid_thw", None)
        if _saved_grid is not None:
            multi_modal_data["image_grid_thw"] = _saved_grid
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})

        # Encode the last tool-generated image as base64 for the image reward judge.
        # Input images are registered under keys like "img_1", "img_2"; tool outputs
        # use names like "ImageGeneration_0", "ImageEdit_0". We only look at the latter.
        _tool_img_keys = [k for k in agent_data.shared_tool_outputs if not k.startswith("img_")]
        _final_image_b64 = None
        if _tool_img_keys:
            _last_img = agent_data.shared_tool_outputs[_tool_img_keys[-1]]
            if _last_img is not None:
                try:
                    _buf = io.BytesIO()
                    _prepare_jpeg_image(_last_img).save(_buf, format="JPEG", quality=85)
                    _final_image_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
                except Exception as _e:
                    logger.warning("Failed to encode final image for reward: %s", _e)
        output.extra_fields["final_image_b64"] = _final_image_b64

        # Encode the first user input image (img_1) for the judge to compare against.
        _input_image_b64 = None
        _input_img = agent_data.shared_tool_outputs.get("img_1")
        if _input_img is not None:
            try:
                _buf = io.BytesIO()
                _prepare_jpeg_image(_input_img).save(_buf, format="JPEG", quality=85)
                _input_image_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
            except Exception as _e:
                logger.warning("Failed to encode input image for reward: %s", _e)
        output.extra_fields["input_image_b64"] = _input_image_b64

        # Rollout progress: log trajectory completion
        _elapsed = (datetime.now() - _start_time).total_seconds()
        _n_tools = len(agent_data.tool_rewards)
        _n_errors = sum(1 for r in agent_data.tool_rewards if r < 0)
        print(
            f"[ROLLOUT] Done   req={request_id[:8]} turns={agent_data.assistant_turns} "
            f"tools={_n_tools} errors={_n_errors} elapsed={_elapsed:.0f}s",
            flush=True,
        )

        # Save complete trajectory trace at checkpoint steps
        if _save_trace and _trace_save_dir:
            self._save_trajectory_trace(agent_data, _trace_save_dir, _sample_index, _rollout_n)

        return output

    def _save_trajectory_trace(self, agent_data: AgentData, trace_save_dir: str, sample_index: int, rollout_n: int = 0):
        """Save complete trajectory (images + messages + metrics) to disk at checkpoint steps."""
        # Use sample_index + rollout_n to avoid collisions when rollout.n > 1
        sample_dir = os.path.join(trace_save_dir, f"sample_{sample_index}_n{rollout_n}")
        try:
            os.makedirs(sample_dir, exist_ok=True)

            # Save all images from shared_tool_outputs (input images + intermediate + final)
            for key, img in agent_data.shared_tool_outputs.items():
                if img is None:
                    continue
                try:
                    # Ensure img is a PIL Image before saving
                    if hasattr(img, "save"):
                        _prepare_jpeg_image(img).save(
                            os.path.join(sample_dir, f"{key}.jpg"), format="JPEG", quality=90
                        )
                    else:
                        logger.warning(f"[TRACE_SAVE] Skipping non-image output {key}: {type(img).__name__}")
                except Exception as e:
                    logger.warning(f"[TRACE_SAVE] Failed to save image {key}: {e}")

            # Serialize messages (strip non-serializable image objects)
            serialized_messages = []
            for msg in agent_data.messages:
                s_msg = {"role": msg.get("role", "")}
                content = msg.get("content", "")
                if isinstance(content, str):
                    s_msg["content"] = content
                elif isinstance(content, list):
                    s_msg["content"] = [
                        item if isinstance(item, dict) and item.get("type") != "image" else {"type": "image"}
                        for item in content
                        if isinstance(item, dict)
                    ]
                else:
                    s_msg["content"] = str(content)[:500]
                serialized_messages.append(s_msg)

            trajectory = {
                "request_id": agent_data.request_id,
                "sample_index": sample_index,
                "rollout_n": rollout_n,
                "assistant_turns": agent_data.assistant_turns,
                "user_turns": agent_data.user_turns,
                "tool_rewards": [float(r) for r in agent_data.tool_rewards],
                "turn_scores": [float(s) for s in agent_data.turn_scores],
                "tool_output_keys": list(agent_data.shared_tool_outputs.keys()),
                "messages": serialized_messages,
            }
            with open(os.path.join(sample_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[TRACE_SAVE] Failed to save trajectory sample_{sample_index}_n{rollout_n}: {e}", flush=True)

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            # Preserve image_grid_thw so agent_loop.py can use the SAME grid
            # that was used to generate input_ids, avoiding token-count mismatches
            # when the processor is called again later.
            _grid = model_inputs.get("image_grid_thw", None)
            agent_data._last_image_grid_thw = _grid.clone() if _grid is not None else None
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # Check if context length exceeds max_model_len before generating
        max_model_len = self.prompt_length + self.response_length
        if len(agent_data.prompt_ids) + 1 >= max_model_len:
            logger.warning(
                f"Context length ({len(agent_data.prompt_ids)}) exceeds max_model_len ({max_model_len}), "
                f"terminating agent loop early."
            )
            return AgentState.TERMINATED

        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Decode assistant response and record in messages for complete trajectory tracking
        _response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        agent_data.messages.append({"role": "assistant", "content": _response_text})

        # Trace: log model response to file for the traced trajectory only
        if agent_data.request_id == ToolAgentLoop._trace_request_id:
            tl = ToolAgentLoop._trace_logger
            tl.info('=' * 80)
            tl.info(f"Turn {agent_data.assistant_turns} | tokens={len(agent_data.response_ids)} | "
                    f"total_tokens={len(agent_data.prompt_ids)}")
            tl.info('=' * 80)
            tl.info(f"Model response:\n{_response_text}")
            tl.info('=' * 80)

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Check for explicit <Terminate> signal in response text
        if content.strip().endswith("<Terminate>"):
            agent_data.tool_calls = []
            print(f"[TERMINATE] Detected <Terminate> tag, forcing termination. req={agent_data.request_id[:8]}", flush=True)
            return AgentState.TERMINATED

        # Handle interaction if needed
        # Note: assistant message already added to agent_data.messages above
        if self.interaction_config_file:
            pass

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        # Execute tools sequentially to support tool output chaining
        # (later tools may depend on earlier tool outputs)
        is_traced = agent_data.request_id == ToolAgentLoop._trace_request_id
        responses = []
        response_output_keys = []  # Track output keys per tool call
        with simple_timer("tool_calls", agent_data.metrics):
            for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
                if is_traced:
                    tl = ToolAgentLoop._trace_logger
                    tl.info(f"Tool call: {tool_call.name}({tool_call.arguments})")
                _t_call_start = time.time()
                tool_response, tool_reward, tool_metrics = await self._call_tool(
                    tool_call,
                    agent_data.tools_kwargs,
                    agent_data.image_data,
                    agent_data.shared_tool_outputs,
                )
                _t_call_elapsed = time.time() - _t_call_start
                if _LOG_TOOL_DISPATCH:
                    _dispatch_extra = ""
                    if tool_call.name in ("ImageEdit", "ImageGeneration"):
                        _gs = tool_metrics.get("guidance_scale", "?")
                        _ns = tool_metrics.get("num_inference_steps", "?")
                        _dispatch_extra = f" guidance_scale={_gs} num_steps={_ns}"
                    print(
                        f"[TOOL_DISPATCH] req={agent_data.request_id[:8]} tool={tool_call.name} "
                        f"turn={agent_data.assistant_turns} elapsed={_t_call_elapsed:.1f}s{_dispatch_extra}",
                        flush=True,
                    )
                # Monitor all tools
                _is_error = bool(tool_response.text and tool_response.text.startswith("Error:"))
                if _is_error:
                    _resp_short = (tool_response.text or "")[:200]
                    _args_short = str(tool_call.arguments)[:150]
                    print(
                        f"[TOOL_MONITOR] {tool_call.name} | req={agent_data.request_id[:8]} "
                        f"turn={agent_data.assistant_turns} | FAIL | "
                        f"reward={tool_reward} elapsed={_t_call_elapsed:.1f}s | "
                        f"args={_args_short} | resp={_resp_short}",
                        flush=True,
                    )
                if is_traced:
                    tl = ToolAgentLoop._trace_logger
                    _text = tool_response.text[:300] + '...' if tool_response.text and len(tool_response.text) > 300 else tool_response.text
                    _has_image = bool(tool_response.image)
                    tl.info(f"Tool response: text={_text}")
                    tl.info(f"Tool response: has_image={_has_image}, reward={tool_reward}")

                # Log trajectory context when tool returns an error
                if tool_response.text and tool_response.text.startswith("Error:"):
                    # Extract user prompt text
                    user_prompt = ""
                    for msg in agent_data.messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                user_prompt = " ".join(
                                    item.get("text", "") for item in content if item.get("type") == "text"
                                )
                            elif isinstance(content, str):
                                user_prompt = content
                    has_image_input = bool(agent_data.image_data)
                    # Decode the current turn's model response directly from response_ids
                    # (agent_data.messages does NOT contain the current assistant response)
                    last_assistant = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
                    print(
                        f"[TOOL_ERROR_CONTEXT] req={agent_data.request_id[:8]} "
                        f"turn={agent_data.assistant_turns} has_image={has_image_input} "
                        f"tool={tool_call.name}({str(tool_call.arguments)[:150]}) "
                        f"| prompt: {user_prompt[:120]} "
                        f"| response: {last_assistant[:300]}",
                        flush=True,
                    )

                responses.append((tool_response, tool_reward, tool_metrics))

                # Store tool output images in shared_tool_outputs for later tools to use
                output_keys = []
                if tool_response.image:
                    tool_name = tool_call.name
                    for idx, img in enumerate(tool_response.image):
                        if img is not None:
                            # Count existing outputs for this tool type
                            existing_count = sum(
                                1 for k in agent_data.shared_tool_outputs.keys()
                                if k.startswith(f"{tool_name}_")
                            )
                            output_key = f"{tool_name}_{existing_count + idx}"
                            agent_data.shared_tool_outputs[output_key] = img
                            output_keys.append(output_key)
                response_output_keys.append(output_keys)

        # Process tool responses and update multi_modal_data
        for (tool_response, tool_reward, _), output_keys in zip(responses, response_output_keys):
            # Build imglist info to append to tool response.
            # Format matches the system prompt: "now imglist is {key1, key2, ...}"
            # Keys include img_1/img_2/... (registered at init for input images)
            # plus tool output keys like Flip_0, ImageGeneration_0, etc.
            imglist_keys = list(agent_data.shared_tool_outputs.keys())
            imglist_str = "{" + ", ".join(imglist_keys) + "}" if imglist_keys else "{}"
            if output_keys:
                img_info = (
                    f"\n[Output image saved as: {', '.join(output_keys)}. "
                    f"now imglist is {imglist_str}]"
                )
            else:
                img_info = f"\n[now imglist is {imglist_str}]"
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                response_text = (tool_response.text or "") + img_info
                content.append({"type": "text", "text": response_text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                response_text = (tool_response.text or "") + img_info
                message = {"role": "tool", "content": response_text}

            add_messages.append(message)
            agent_data.messages.extend(add_messages)

            # Handle image data
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            # Append new image_grid_thw entries from this tool response to the
            # accumulated grid so agent_loop.py has the complete grid for ALL
            # images across all turns (initial prompt + tool responses).
            _new_grid = model_inputs.get("image_grid_thw", None)
            if _new_grid is not None and len(_new_grid) > 0:
                if agent_data._last_image_grid_thw is not None:
                    agent_data._last_image_grid_thw = torch.cat(
                        [agent_data._last_image_grid_thw, _new_grid], dim=0
                    )
                else:
                    agent_data._last_image_grid_thw = _new_grid.clone()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        shared_tool_outputs: Optional[dict[str, Any]] = None,
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Args:
            tool_call: The function call to execute
            tools_kwargs: Tool-specific kwargs from dataset
            image_data: List of all images (initial + tool outputs)
            shared_tool_outputs: Dict mapping output keys to images from previous tools
        """
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            # Pass image_data and shared_tool_outputs to tool execute
            _t_tool_start = time.time()
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id,
                tool_args,
                image_data=image_data,
                shared_tool_outputs=shared_tool_outputs or {},
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                -0.1,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
