"""Strict alignment of Terminus-2 messages, raw ATIF and native reasoning."""

from __future__ import annotations

import re

ERROR_PATTERN = re.compile(
    r"command not found|No such file or directory|Traceback \(most recent call last\)|"
    r"fatal error:|undefined reference to|Previous response had parsing errors:",
    re.I,
)
ANSWER_PATTERN = re.compile(
    r"/var/lib/(?:docker|containerd)|/proc/(?:self|[0-9]+)/mountinfo|"
    r"/opt/(?:reference|LAPS-straw)|/tests/reference|/solution/(?:defect_fix|solve\.sh)|"
    r"git\s+(?:clone|fetch|pull)\b",
    re.I,
)


def align_episode(result, atif):
    """Strict raw ATIF only: refuse count repair, summaries and fabricated CoT."""
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    if atif.get("continued_trajectory_ref") or atif.get("subagent_trajectories"):
        raise ValueError("unsupported_continuation_or_summary")
    all_steps = atif.get("steps", [])
    if any(s.get("is_copied_context") for s in all_steps):
        raise ValueError("copied_context")
    if any(
        s.get("source") not in {"user", "agent"}
        or any(r.get("subagent_trajectory_ref") for r in (s.get("observation") or {}).get("results", []))
        for s in all_steps
    ):
        raise ValueError("unsupported_compaction_or_subagent")
    steps = [s for s in all_steps if s.get("source") == "agent"]
    messages = (result.get("agent_result") or {}).get("metadata", {}).get("all_messages")
    if not isinstance(messages, list) or len(steps) < 2 or len(messages) != 2 * len(steps):
        raise ValueError("raw_atif_turn_count_mismatch")
    initial = [s for s in all_steps if s.get("source") == "user"]
    if len(initial) != 1 or initial[0].get("message") != messages[0].get("content"):
        raise ValueError("initial_user_mismatch")
    parser = TerminusJSONPlainParser()
    full, parsed, masks = [], [], []
    command_targets = 0
    for index, step in enumerate(steps):
        user, assistant = messages[2 * index : 2 * index + 2]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("unexpected_dialogue_role")
        if not isinstance(user.get("content"), str) or not isinstance(assistant.get("content"), str):
            raise ValueError("non_text_dialogue")
        if assistant["content"] != step.get("message"):
            raise ValueError("raw_atif_content_mismatch")
        feedback = (step.get("observation") or {}).get("results", [])
        if len(feedback) != 1 or not isinstance(feedback[0].get("content"), str):
            raise ValueError("unsupported_terminal_observation")
        observation = feedback[0]["content"]
        if index + 1 < len(steps) and observation != messages[2 * index + 2]["content"]:
            raise ValueError("terminal_feedback_mismatch")
        response = parser.parse_response(assistant["content"])
        parsed.append(response)
        command_text = "\n".join(c.keystrokes for c in response.commands)
        if ANSWER_PATTERN.search(command_text):
            raise ValueError("answer_material_or_upstream_access")
        reasoning = step.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("non_text_native_reasoning")
        reason = None
        if response.error or response.warning:
            reason = "protocol_error_or_warning_context_only"
        elif not reasoning or not reasoning.strip():
            reason = "missing_native_reasoning_context_only"
        elif ERROR_PATTERN.search(observation):
            reason = "explicit_terminal_error_context_only"
        loss = reason is None
        command_targets += int(loss and bool(response.commands))
        full.extend(
            [
                {"role": "user", "content": user["content"], "loss": False},
                {
                    "role": "assistant",
                    "content": assistant["content"],
                    "reasoning_content": reasoning or "",
                    "loss": loss,
                },
            ]
        )
        masks.append({"turn": index + 1, "loss": loss, "reason": reason})
    if not all(p.is_task_complete and not p.error for p in parsed[-2:]):
        raise ValueError("missing_completion_confirmation")
    if sum(m["loss"] for m in masks) < 2 or command_targets == 0:
        raise ValueError("insufficient_native_reasoning_action_targets")
    return full, masks
