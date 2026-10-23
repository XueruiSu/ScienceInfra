"""Offline contract tests; no test launches Docker or calls a model endpoint.

Run with the Python environment containing the pinned Harbor and pandas:
    python -m unittest discover -s tests/sft -p 'test_*.py' -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

# Importing Harbor's parser imports LiteLLM through its package. Prevent the
# library's optional model-price refresh so this suite stays entirely offline.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


from scienceinfra.sft import rollout, tokenize_qwen

CHECKS = ["aw-2d-256", "aw-2d-512"]
SOURCE = "src_compressible/2D/mhdrhs.f90"
FIX_LINE = 'python3 "$HERE/apply_defect.py" "$HERE/defect_fix.json" /app/LAPS'
MANIFEST = """schema_version = "1.4"
[task]
name = "sciaccel/fixture-repair"
[metadata.taxonomy]
category = "repair"
tree = "2d"
affected_checks = ["aw-2d-256", "aw-2d-512"]
floor_native_estimate = 0.5
[environment]
cpus = 4
memory_mb = 4096
network_mode = "allowlist"
allowed_hosts = ["github.com", "example.com"]
[agent]
timeout_sec = 3600.0
network_mode = "allowlist"
allowed_hosts = ["api.example.com"]
[verifier]
environment_mode = "separate"
timeout_sec = 900.0
[verifier.environment]
network_mode = "no-network"
cpus = 4
"""


def response(commands=None, complete=False):
    return json.dumps(
        {
            "analysis": "Inspect or repair the numerical source.",
            "plan": "Compile and validate the required cases.",
            "commands": commands or [],
            "task_complete": complete,
        }
    )


def dialogue(reasoning=None):
    """Real Terminus raw-content shape, including two completion confirmations."""
    contents = [
        response([{"keystrokes": "sed -n '1,20p' /app/LAPS/src_compressible/2D/mhdrhs.f90\n", "duration": 0.1}]),
        response(complete=True),
        response(complete=True),
    ]
    prompts = [
        "Task, required cases and initial terminal state",
        "Current Terminal Screen:\nsource inspected",
        "Completion confirmation request",
    ]
    thoughts = reasoning or ["Inspect the source.", "Check the delivered frames.", "The task is finished."]
    messages, steps = [], [{"step_id": 1, "source": "user", "message": prompts[0]}]
    for index, content in enumerate(contents):
        messages.extend([{"role": "user", "content": prompts[index]}, {"role": "assistant", "content": content}])
        steps.append(
            {
                "step_id": index + 2,
                "source": "agent",
                "message": content,
                "reasoning_content": thoughts[index],
                "observation": {
                    "results": [
                        {
                            "content": prompts[index + 1]
                            if index + 1 < len(prompts)
                            else "Final terminal evidence, never sent to model"
                        }
                    ]
                },
            }
        )
    result = {"agent_result": {"metadata": {"all_messages": messages}}}
    return result, {"steps": steps}


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = {
            "id": "fixture",
            "task_name": "sciaccel/fixture-repair",
            "path": "tasks/fixture/task",
            "checks": CHECKS.copy(),
            "hint": "Look at the numerical source.",
            "hint_level": "L1",
        }
        task_path = self.root / self.task["path"]
        (task_path / "environment").mkdir(parents=True)
        (task_path / "environment/Dockerfile").write_text("FROM scratch\n")
        (task_path / "instruction.md").write_text("Repair the solver and deliver all declared checks.\n")
        (task_path / "task.toml").write_text(MANIFEST)
        solution = task_path / "solution"
        solution.mkdir()
        (solution / "apply_defect.py").write_text("raise RuntimeError('official fix must never run during replay')\n")
        (solution / "defect_fix.json").write_text('{"official": "answer"}\n')
        (solution / "solve.sh").write_text(
            '#!/bin/bash\nset -euo pipefail\nHERE="$(dirname "$0")"\n'
            + FIX_LINE
            + "\nmake -C /app/LAPS/src_compressible/2D\n"
        )
        self.plan = {
            "schema": rollout.SCHEMA,
            "teacher_model": "GLM-5.3",
            "api_base": "https://api.example.test/v1",
            "source_root": "/app/LAPS",
            "source_replay_adapter": "laps-pinned-solve-v1",
            "max_turns": 64,
            "max_output_tokens": 8192,
            "context_tokens": 65536,
            "temperature": 0.6,
            "top_p": 0.95,
            "concurrency": 2,
            "attempts_per_task": 4,
            "tasks": [self.task],
            "runtime": rollout.runtime_record(),
        }
        rollout.write(self.root / "plan.json", self.plan)

    def trial(self, folder="trial", reward=1.0, floor=0.5, **changes):
        path = self.root / folder / "result.json"
        rewards = {
            "reward": reward,
            "floor": floor,
            "reward_repair": round(max(0, reward - floor) / max(1e-6, 1 - floor), 6),
            "equivalence_pass": int(reward == 1),
            "speedup": 0.0,
            **{"check_" + c.replace("-", "_"): reward for c in CHECKS},
        }
        rewards.update(changes)
        result = {
            "task_name": self.task["task_name"],
            "task_checksum": "fixture-checksum",
            "finished_at": "2026-10-22T00:00:00Z",
            "exception_info": None,
            "verifier_result": {"rewards": rewards},
            "agent_result": {"metadata": {}},
        }
        rollout.write(path, result)
        rollout.write(path.parent / "verifier/reward.json", rewards)
        rollout.write(
            path.parent / "verifier/grading_report.json",
            {
                "checks": {
                    c: {
                        "passed": reward == 1,
                        "outcome": "pass" if reward == 1 else "mismatch",
                        "frames_ok": 10,
                        "nstep_match": True,
                        "dt_abs": 0.0,
                        "time_abs": 0.0,
                        "time_bound": 1e-10,
                    }
                    for c in CHECKS
                }
            },
        )
        return path

    def captured_source(self, result_path, text="rhs = rhs + flux\n", extra=None):
        path = result_path.parent / "artifacts/app/LAPS"
        (path / SOURCE).parent.mkdir(parents=True, exist_ok=True)
        (path / SOURCE).write_text(text)
        for rel, content in (extra or {}).items():
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            (path / rel).write_text(content)
        return path

    def anchor(self):
        oracle = self.trial("anchors/fixture/oracle/jobs/job/oracle")
        nop = self.trial("anchors/fixture/nop/jobs/job/nop", reward=0.5)
        self.captured_source(nop, "rhs = rhs - flux\n")
        receipt = {
            "status": "passed",
            "floor": 0.5,
            "oracle_result": str(oracle.relative_to(self.root)),
            "oracle_result_sha256": rollout.digest(oracle),
            "nop_result": str(nop.relative_to(self.root)),
            "nop_result_sha256": rollout.digest(nop),
            "baseline_sources": rollout.source_files(nop.parent / "artifacts/app/LAPS"),
            "evidence_sha256": {},
        }
        rollout.write(self.root / "anchors/fixture/anchor.json", receipt)
        return receipt


class ScientificRewardTests(FixtureCase):
    def test_task_specific_two_dimensional_checks_do_not_require_three_dimensional_case(self):
        path = self.trial()
        _, reward = rollout.scientific_pass(path, self.task, 0.5)
        self.assertEqual(reward["reward_repair"], 1)
        self.assertNotIn("check_aw_128", reward)

    def test_partial_raw_reward_is_not_a_success(self):
        with self.assertRaisesRegex(ValueError, "scientific_score_below_one"):
            rollout.scientific_pass(self.trial(reward=0.75), self.task)

    def test_full_reward_for_another_task_cannot_be_attributed_to_the_planned_task(self):
        path = self.trial()
        result = rollout.read(path)
        result["task_name"] = "sciaccel/different-task-with-the-same-checks"
        rollout.write(path, result)
        with self.assertRaisesRegex(ValueError, "trial_task_name_mismatch"):
            rollout.scientific_pass(path, self.task)

    def test_measured_zero_is_valid_but_an_empty_measurement_is_rejected(self):
        path = self.trial(reward=0.0)
        self.assertEqual(rollout.reward_record(path)[1]["reward_repair"], 0)
        rollout.write(path.parent / "verifier/reward.json", {})
        data = rollout.read(path)
        data["verifier_result"]["rewards"] = {}
        rollout.write(path, data)
        with self.assertRaisesRegex(ValueError, "missing_numeric_reward"):
            rollout.reward_record(path)

    def test_reward_result_disagreement_is_rejected(self):
        path = self.trial()
        payload = rollout.read(path.parent / "verifier/reward.json")
        payload["floor"] = 0.4
        rollout.write(path.parent / "verifier/reward.json", payload)
        with self.assertRaisesRegex(ValueError, "reward_result_mismatch"):
            rollout.reward_record(path)

    def test_floor_drift_and_inconsistent_normalization_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "measured_floor_drift"):
            rollout.scientific_pass(self.trial(), self.task, 0.25)
        with self.assertRaisesRegex(ValueError, "repair_score_inconsistent"):
            rollout.reward_record(self.trial("inconsistent", reward_repair=0.8))

    def test_invalid_or_missing_numeric_rewards_are_rejected(self):
        for value in [None, True, "1", float("nan"), float("inf")]:
            with self.subTest(value=value):
                path = self.trial("bad-number")
                result = rollout.read(path)
                result["verifier_result"]["rewards"]["reward"] = value
                rollout.write(path, result)
                rollout.write(path.parent / "verifier/reward.json", result["verifier_result"]["rewards"])
                with self.assertRaisesRegex(ValueError, "missing_numeric_reward"):
                    rollout.reward_record(path)

    def test_invalid_floors_are_rejected(self):
        for floor in [-0.01, 1.0, 1.1]:
            with self.subTest(floor=floor), self.assertRaisesRegex(ValueError, "invalid_measured_floor"):
                rollout.reward_record(self.trial("bad-floor", floor=floor))

    def test_grader_crash_and_runtime_exception_are_not_model_failures(self):
        with self.assertRaisesRegex(ValueError, "grader_error"):
            rollout.reward_record(self.trial(grader_error=1))
        path = self.trial("runtime")
        data = rollout.read(path)
        data["exception_info"] = {"exception_type": "VerifierTimeoutError"}
        rollout.write(path, data)
        with self.assertRaisesRegex(ValueError, "runtime_exception"):
            rollout.reward_record(path)

    def test_required_check_and_strict_grading_verdict_are_both_required(self):
        path = self.trial(check_aw_2d_512=0.0)
        with self.assertRaisesRegex(ValueError, "required_check_failed:aw-2d-512"):
            rollout.scientific_pass(path, self.task)
        path = self.trial("report-failed")
        report_path = path.parent / "verifier/grading_report.json"
        report = rollout.read(report_path)
        report["checks"]["aw-2d-512"] = {"passed": False, "outcome": "grader_exception"}
        rollout.write(report_path, report)
        with self.assertRaisesRegex(ValueError, "grading_report_failed:aw-2d-512"):
            rollout.scientific_pass(path, self.task)

    def test_anchor_receipt_authenticates_raw_trial_and_floor(self):
        self.anchor()
        self.assertEqual(rollout.verify_anchor(self.root, self.task)["floor"], 0.5)
        path = self.root / rollout.read(self.root / "anchors/fixture/anchor.json")["nop_result"]
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Anchor raw result changed"):
            rollout.verify_anchor(self.root, self.task)

    def test_existing_nop_anchor_must_belong_to_the_planned_task(self):
        receipt = self.anchor()
        path = self.root / receipt["nop_result"]
        result = rollout.read(path)
        result["task_name"] = "sciaccel/other-nop-task"
        rollout.write(path, result)
        # Even a self-consistent receipt must not disguise a different task.
        receipt["nop_result_sha256"] = rollout.digest(path)
        rollout.write(self.root / "anchors/fixture/anchor.json", receipt)
        with self.assertRaisesRegex(ValueError, "nop_task_name_mismatch"):
            rollout.verify_anchor(self.root, self.task)

    def test_new_anchor_rejects_a_nop_result_for_a_different_task(self):
        oracle = self.trial("new-oracle")
        nop = self.trial("new-nop", reward=0.5)
        result = rollout.read(nop)
        result["task_name"] = "sciaccel/different-nop"
        rollout.write(nop, result)
        with (
            mock.patch.object(rollout, "execute_job", side_effect=[oracle, nop]),
            self.assertRaisesRegex(ValueError, "nop_task_name_mismatch"),
        ):
            rollout.anchors(self.root, self.plan)
        self.assertFalse((self.root / "anchors/fixture/anchor.json").exists())


class RawDialogueTests(unittest.TestCase):
    def test_exact_raw_content_and_native_reasoning_are_preserved(self):
        result, atif = dialogue()
        messages, masks = rollout.align_episode(result, atif)
        self.assertEqual(len(messages), 6)
        self.assertEqual([m["loss"] for m in messages], [False, True] * 3)
        self.assertEqual([m["loss"] for m in masks], [True] * 3)
        for index, step in enumerate(atif["steps"][1:]):
            self.assertEqual(messages[2 * index + 1]["content"], step["message"])
            self.assertEqual(messages[2 * index + 1]["reasoning_content"], step["reasoning_content"])
        self.assertEqual(len(result["agent_result"]["metadata"]["all_messages"]), 6)
        self.assertNotIn("reasoning_content", result["agent_result"]["metadata"]["all_messages"][1])

    def test_missing_native_reasoning_is_context_only_without_invented_thought(self):
        result, atif = dialogue(["Inspect the source.", None, "Confirm completion."])
        messages, masks = rollout.align_episode(result, atif)
        self.assertFalse(messages[3]["loss"])
        self.assertEqual(messages[3]["reasoning_content"], "")
        self.assertEqual(masks[1]["reason"], "missing_native_reasoning_context_only")

    def test_count_mismatch_is_rejected_instead_of_dropping_turns(self):
        result, atif = dialogue()
        result["agent_result"]["metadata"]["all_messages"].extend(
            [{"role": "user", "content": "retry"}, {"role": "assistant", "content": "truncated"}]
        )
        with self.assertRaisesRegex(ValueError, "raw_atif_turn_count_mismatch"):
            rollout.align_episode(result, atif)

    def test_raw_assistant_or_terminal_feedback_mismatch_is_rejected(self):
        for kind in ["assistant", "feedback", "initial"]:
            result, atif = dialogue()
            index = {"assistant": 1, "feedback": 2, "initial": 0}[kind]
            result["agent_result"]["metadata"]["all_messages"][index]["content"] += " tampered"
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(
                    ValueError,
                    {
                        "assistant": "raw_atif_content_mismatch",
                        "feedback": "terminal_feedback_mismatch",
                        "initial": "initial_user_mismatch",
                    }[kind],
                ),
            ):
                rollout.align_episode(result, atif)

    def test_one_completion_claim_is_not_confirmed_completion(self):
        result, atif = dialogue()
        updated = response(complete=False)
        result["agent_result"]["metadata"]["all_messages"][3]["content"] = updated
        atif["steps"][2]["message"] = updated
        with self.assertRaisesRegex(ValueError, "missing_completion_confirmation"):
            rollout.align_episode(result, atif)

    def test_explicit_terminal_errors_are_context_only(self):
        result, atif = dialogue()
        error = "Current Terminal Screen:\nTraceback (most recent call last): temporary check failed"
        atif["steps"][2]["observation"]["results"][0]["content"] = error
        result["agent_result"]["metadata"]["all_messages"][4]["content"] = error
        _, masks = rollout.align_episode(result, atif)
        self.assertEqual(masks[1]["reason"], "explicit_terminal_error_context_only")
        self.assertFalse(masks[1]["loss"])

    def test_parser_warning_is_not_supervised_but_original_content_is_retained(self):
        result, atif = dialogue()
        raw = "Extra text tolerated by parser\n" + atif["steps"][2]["message"]
        result["agent_result"]["metadata"]["all_messages"][3]["content"] = raw
        atif["steps"][2]["message"] = raw
        messages, masks = rollout.align_episode(result, atif)
        self.assertEqual(messages[3]["content"], raw)
        self.assertEqual(masks[1]["reason"], "protocol_error_or_warning_context_only")

    def test_malformed_protocol_is_context_only_without_inventing_executed_commands(self):
        result, atif = dialogue()
        malformed = "This is not a JSON command response."
        result["agent_result"]["metadata"]["all_messages"][1]["content"] = malformed
        atif["steps"][1]["message"] = malformed
        # A later, valid action completes the repair, followed by confirmation.
        valid = response([{"keystrokes": "make -C /app/LAPS/src_compressible/2D\n", "duration": 1}], complete=True)
        result["agent_result"]["metadata"]["all_messages"][3]["content"] = valid
        atif["steps"][2]["message"] = valid
        messages, masks = rollout.align_episode(result, atif)
        self.assertEqual(messages[1]["content"], malformed)
        self.assertFalse(messages[1]["loss"])
        self.assertEqual(masks[0]["reason"], "protocol_error_or_warning_context_only")
        self.assertNotIn("tool_calls", messages[1])

    def test_no_native_reasoning_or_no_action_targets_is_rejected(self):
        result, atif = dialogue([None, None, None])
        with self.assertRaisesRegex(ValueError, "insufficient_native_reasoning_action_targets"):
            rollout.align_episode(result, atif)
        result, atif = dialogue()
        raw = response()
        result["agent_result"]["metadata"]["all_messages"][1]["content"] = raw
        atif["steps"][1]["message"] = raw
        with self.assertRaisesRegex(ValueError, "insufficient_native_reasoning_action_targets"):
            rollout.align_episode(result, atif)

    def test_answer_paths_and_upstream_fetch_commands_are_rejected(self):
        for command in [
            "cat /opt/reference/answer\n",
            "cat /tests/reference/aw-128/out001.dat\n",
            "cat /solution/defect_fix.json\n",
            "cat /solution/solve.sh\n",
            "git clone https://github.com/example/LAPS\n",
            "cat /proc/self/mountinfo\n",
        ]:
            result, atif = dialogue()
            raw = response([{"keystrokes": command, "duration": 0.1}])
            result["agent_result"]["metadata"]["all_messages"][1]["content"] = raw
            atif["steps"][1]["message"] = raw
            with (
                self.subTest(command=command),
                self.assertRaisesRegex(ValueError, "answer_material_or_upstream_access"),
            ):
                rollout.align_episode(result, atif)

    def test_non_text_roles_copied_context_and_continuations_are_rejected(self):
        mutations = [
            (
                lambda r, a: r["agent_result"]["metadata"]["all_messages"][1].update(role="tool"),
                "unexpected_dialogue_role",
            ),
            (lambda r, a: a["steps"][1].update(reasoning_content=["not text"]), "non_text_native_reasoning"),
            (lambda r, a: a["steps"][1].update(is_copied_context=True), "copied_context"),
            (
                lambda r, a: a.update(continued_trajectory_ref="trajectory.cont-1.json"),
                "unsupported_continuation_or_summary",
            ),
            (lambda r, a: a["steps"][1].update(observation={"results": []}), "unsupported_terminal_observation"),
        ]
        for mutate, error in mutations:
            result, atif = dialogue()
            mutate(result, atif)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                rollout.align_episode(result, atif)

    def test_compaction_or_subagent_observations_are_rejected_even_with_matching_turn_counts(self):
        for kind in ["compaction", "subagent"]:
            result, atif = dialogue()
            if kind == "compaction":
                atif["steps"].insert(
                    1,
                    {
                        "step_id": 2,
                        "source": "system",
                        "message": "Context compacted",
                        "extra": {"context_management": {"type": "compaction", "boundary": "replace"}},
                    },
                )
            else:
                atif["steps"][1]["observation"]["results"][0]["subagent_trajectory_ref"] = [
                    {"trajectory_path": "subagent.json"}
                ]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unsupported_compaction_or_subagent"):
                rollout.align_episode(result, atif)


class StagingAndEvidenceTests(FixtureCase):
    def test_init_cli_requires_an_explicit_api_endpoint(self):
        argv = ["init", "--dataset", "fixture.parquet", "--run-dir", "fixture-run"]
        error = io.StringIO()
        with mock.patch.object(rollout, "init_run") as init_run:
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                rollout.main(argv)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--api-base", error.getvalue())
            init_run.assert_not_called()

            endpoint = "https://api.example.test/v1"
            rollout.main([*argv, "--api-base", endpoint])
            self.assertEqual(init_run.call_args.args[0].api_base, endpoint)
            self.assertEqual(init_run.call_args.args[0].model, "GLM-5.3")

            rollout.main([*argv, "--api-base", endpoint, "--model", "custom-model"])
            self.assertEqual(init_run.call_args.args[0].model, "custom-model")

    def test_runtime_network_staging_preserves_scientific_and_verifier_fields(self):
        changed = tomllib.loads(rollout.no_network(MANIFEST))
        before = tomllib.loads(MANIFEST)
        for section in ["environment", "agent"]:
            self.assertEqual(changed[section]["network_mode"], "no-network")
            self.assertEqual(changed[section]["allowed_hosts"], [])
            for data in [before, changed]:
                data[section].pop("network_mode")
                data[section].pop("allowed_hosts")
        self.assertEqual(before, changed)

    def test_network_staging_handles_standard_multiline_allowed_hosts(self):
        manifest = MANIFEST.replace('["github.com", "example.com"]', '[\n  "github.com",\n  "example.com",\n]')
        changed = tomllib.loads(rollout.no_network(manifest))
        self.assertEqual(changed["environment"]["allowed_hosts"], [])
        self.assertEqual(changed["metadata"]["taxonomy"]["affected_checks"], CHECKS)

    def test_plan_digest_and_staged_evidence_changes_are_rejected(self):
        self.task["task_files"] = rollout.files(self.root / self.task["path"])
        rollout.write(self.root / "plan.json", self.plan)
        (self.root / "plan.sha256").write_text(rollout.digest(self.root / "plan.json"))
        self.assertEqual(rollout.load_plan(self.root)["tasks"][0]["task_name"], self.task["task_name"])
        original = (self.root / "plan.json").read_text()
        (self.root / "plan.json").write_text(original + " ")
        with self.assertRaisesRegex(ValueError, "Run plan changed"):
            rollout.load_plan(self.root)
        (self.root / "plan.json").write_text(original)
        (self.root / self.task["path"] / "instruction.md").write_text("changed task")
        with self.assertRaisesRegex(ValueError, "Staged task changed"):
            rollout.load_plan(self.root)

    def test_plan_rejects_using_changed_starter_code(self):
        self.task["task_files"] = rollout.files(self.root / self.task["path"])
        self.plan["runtime"]["starter_sha256"] = "not-the-running-code"
        rollout.write(self.root / "plan.json", self.plan)
        (self.root / "plan.sha256").write_text(rollout.digest(self.root / "plan.json"))
        with self.assertRaisesRegex(ValueError, "Starter code changed"):
            rollout.load_plan(self.root)

    def test_init_stages_a_fresh_task_and_rejects_reusing_run_directory(self):
        dataset = self.root / "input.parquet"
        dataset.write_bytes(b"fixture parquet input: reader is mocked")
        frame = mock.Mock()
        frame.to_dict.return_value = [
            {
                "task_name": self.task["task_name"],
                "floor": 0.5,
                "extra_info": {
                    "task_path": str(self.root / self.task["path"]),
                    "reward_key": "reward_repair",
                    "gpus": 0,
                    "hint": "L1 hint",
                    "hint_level": "L1",
                },
            }
        ]
        args = argparse.Namespace(
            run_dir=self.root / "fresh",
            dataset=dataset,
            attempts=4,
            concurrency=2,
            limit=1,
            api_base="https://api.example.test/v1",
            model="GLM-5.3",
            max_turns=64,
            max_output_tokens=8192,
            context_tokens=65536,
        )
        with (
            mock.patch.dict(sys.modules, {"pandas": mock.Mock(read_parquet=mock.Mock(return_value=frame))}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rollout.init_run(args)
        plan = rollout.load_plan(args.run_dir)
        self.assertEqual(plan["attempts_per_task"], 4)
        self.assertEqual(plan["tasks"][0]["hint"], "L1 hint")
        self.assertNotIn("api_key", json.dumps(plan).lower())
        self.assertEqual(
            tomllib.loads((args.run_dir / plan["tasks"][0]["path"] / "task.toml").read_text())["environment"][
                "network_mode"
            ],
            "no-network",
        )
        with (
            mock.patch.dict(sys.modules, {"pandas": mock.Mock()}),
            self.assertRaisesRegex(ValueError, "NEW run directory"),
        ):
            rollout.init_run(args)

    def test_init_rejects_a_parquet_task_name_that_disagrees_with_the_manifest(self):
        dataset = self.root / "wrong-task.parquet"
        dataset.write_bytes(b"mocked task input")
        frame = mock.Mock()
        frame.to_dict.return_value = [
            {
                "task_name": "sciaccel/wrong-parquet-name",
                "floor": 0.5,
                "extra_info": {
                    "task_path": str(self.root / self.task["path"]),
                    "reward_key": "reward_repair",
                    "gpus": 0,
                },
            }
        ]
        args = argparse.Namespace(
            run_dir=self.root / "wrong-run",
            dataset=dataset,
            attempts=4,
            concurrency=2,
            limit=1,
            api_base="https://api.example.test/v1",
            model="GLM-5.3",
            max_turns=64,
            max_output_tokens=8192,
            context_tokens=65536,
        )
        with (
            mock.patch.dict(sys.modules, {"pandas": mock.Mock(read_parquet=mock.Mock(return_value=frame))}),
            self.assertRaisesRegex(ValueError, "task_name does not match"),
        ):
            rollout.init_run(args)
        self.assertFalse((args.run_dir / "plan.json").exists())

    def test_reviewed_evidence_tampering_is_rejected(self):
        folder = self.root / "attempts/fixture/attempt-001"
        evidence = folder / "evidence.txt"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("original evidence")
        rollout.write(
            folder / "review.json",
            {"status": "accepted", "evidence_sha256": {str(evidence.relative_to(self.root)): rollout.digest(evidence)}},
        )
        self.assertEqual(rollout.review_one(self.root, self.plan, self.task, 1, folder)["status"], "accepted")
        evidence.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Reviewed evidence changed"):
            rollout.review_one(self.root, self.plan, self.task, 1, folder)

    def test_export_with_no_accepted_episode_reports_empty_instead_of_fabricating_sft(self):
        output = self.root / "release"
        with contextlib.redirect_stdout(io.StringIO()):
            code = rollout.export_release(self.root, self.plan, output)
        self.assertEqual(code, 2)
        self.assertEqual((output / "sft/train.jsonl").read_text(), "")
        manifest = rollout.read(output / "manifest.json")
        self.assertEqual(manifest["status"], "no_admissible_episodes")
        self.assertFalse(manifest["model_training_performed"])
        self.assertEqual(rollout.read(output / "statistics.json")["planned_attempts"], 4)
        for rel, expected in manifest["files"].items():
            self.assertEqual(rollout.digest(output / rel), expected)
        with self.assertRaisesRegex(ValueError, "NEW directory"):
            rollout.export_release(self.root, self.plan, output)

    def test_job_argv_preserves_hint_raw_content_and_hard_output_limit_without_key(self):
        folder = self.root / "attempts/fixture/attempt-001"
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-secret"}):
            command = rollout.job_command(self.root, self.plan, self.task, folder, "terminus-2")
        self.assertNotIn("fixture-secret", " ".join(command))
        self.assertEqual(command[command.index("--extra-instruction") + 1], self.task["hint"])
        options = dict(command[i + 1].split("=", 1) for i, value in enumerate(command) if value == "--ak")
        self.assertTrue(json.loads(options["store_all_messages"]))
        self.assertFalse(json.loads(options["enable_summarize"]))
        self.assertEqual(json.loads(options["trajectory_config"]), {"raw_content": True})
        self.assertEqual(json.loads(options["llm_call_kwargs"])["max_tokens"], 8192)


class SourceReplayTests(FixtureCase):
    def candidate(self, text="rhs = rhs + flux\n", extra=None):
        self.anchor()
        folder = self.root / "attempts/fixture/attempt-001"
        path = self.trial("attempts/fixture/attempt-001/jobs/job/candidate")
        self.captured_source(path, text, extra)
        return folder, path

    def test_unchanged_source_is_rejected_even_with_full_scientific_score(self):
        folder, path = self.candidate("rhs = rhs - flux\n")
        with self.assertRaisesRegex(ValueError, "no_scientific_source_change"):
            rollout.build_replay(self.root, self.plan, self.task, path, folder)

    def test_missing_source_or_added_source_file_requires_explicit_adapter_support(self):
        with self.assertRaisesRegex(ValueError, "missing_captured_source"):
            rollout.source_files(self.root / "absent")
        folder, path = self.candidate(extra={"src_compressible/2D/invented.f90": "new source"})
        with self.assertRaisesRegex(ValueError, "source_file_set_changed_not_supported"):
            rollout.build_replay(self.root, self.plan, self.task, path, folder)

    def test_captured_source_symlinks_cannot_escape_or_alias_evidence(self):
        directory = self.root / "captured"
        directory.mkdir()
        outside = self.root / "outside.f90"
        outside.write_text("source outside the capture")
        (directory / "aliased.f90").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink not allowed"):
            rollout.source_files(directory)

    def test_replay_removes_official_fix_and_injects_only_recorded_scientific_source(self):
        folder, path = self.candidate(
            extra={"src_compressible/2D/mhd.exe": "stale binary", "notes.txt": "not scientific source"}
        )
        replay = rollout.build_replay(self.root, self.plan, self.task, path, folder)
        script = (replay / "solution/solve.sh").read_text()
        self.assertNotIn("apply_defect.py", script)
        self.assertNotIn("defect_fix.json", script)
        self.assertFalse((replay / "solution/defect_fix.json").exists())
        self.assertFalse((replay / "solution/apply_defect.py").exists())
        self.assertEqual(
            rollout.files(replay / "solution/sampled_source"),
            {SOURCE: rollout.digest(path.parent / "artifacts/app/LAPS" / SOURCE)},
        )
        injector = (replay / "solution/replay_source.py").read_text()
        compile(injector, "replay_source.py", "exec")
        self.assertIn('".exe"', injector)
        self.assertIn("p.unlink()", injector)
        self.assertIn("baseline source drift", injector)
        self.assertEqual(rollout.read(folder / "source_changes.json")["changed_sources"], [SOURCE])
        self.assertEqual(rollout.build_replay(self.root, self.plan, self.task, path, folder), replay)

    def test_replay_rejects_changed_captured_candidate_source(self):
        folder, path = self.candidate()
        rollout.build_replay(self.root, self.plan, self.task, path, folder)
        (path.parent / "artifacts/app/LAPS" / SOURCE).write_text("changed after replay preparation")
        with self.assertRaisesRegex(ValueError, "Source replay inputs changed"):
            rollout.build_replay(self.root, self.plan, self.task, path, folder)

    def test_replay_rejects_changes_to_prepared_injector_or_sampled_source(self):
        folder, path = self.candidate()
        replay = rollout.build_replay(self.root, self.plan, self.task, path, folder)
        (replay / "solution/replay_source.py").write_text("print('tampered replay')\n")
        with self.assertRaisesRegex(ValueError, "Prepared replay task changed"):
            rollout.build_replay(self.root, self.plan, self.task, path, folder)

    def test_anchor_authenticates_captured_defective_baseline_source(self):
        receipt = self.anchor()
        baseline = (self.root / receipt["nop_result"]).parent / "artifacts/app/LAPS" / SOURCE
        baseline.write_text("silently changed baseline")
        with self.assertRaisesRegex(ValueError, "Captured defective baseline source changed"):
            rollout.verify_anchor(self.root, self.task)


class ExportAdmissionTests(FixtureCase):
    def accepted_fixture(self):
        """Exercise real review/export; only the CPU replay process is mocked."""
        self.plan["attempts_per_task"] = 1
        rollout.write(self.root / "plan.json", self.plan)
        anchor = self.anchor()
        folder = self.root / "attempts/fixture/attempt-001"
        path = self.trial("attempts/fixture/attempt-001/jobs/job/trial")
        result = rollout.read(path)
        conversation, atif = dialogue()
        result["agent_result"] = conversation["agent_result"]
        result["agent_result"].update(n_input_tokens=12000, n_output_tokens=1000)
        rollout.write(path, result)
        rollout.write(path.parent / "agent/trajectory.json", atif)
        self.captured_source(path)
        rollout.write(
            folder / "execution.json",
            {
                "status": "finished",
                "agent": "terminus-2",
                "result": str(path.relative_to(self.root)),
                "result_sha256": rollout.digest(path),
                "plan_sha256": rollout.digest(self.root / "plan.json"),
                "task_files": rollout.files(self.root / self.task["path"]),
            },
        )
        replay_result = self.trial("attempts/fixture/attempt-001/source-replay/jobs/job/replay")
        with mock.patch.object(rollout, "execute_job", return_value=replay_result) as execute:
            review = rollout.review_one(self.root, self.plan, self.task, 1, folder)
        self.assertEqual(review["status"], "accepted", review)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[4], "oracle")
        return folder, path, anchor

    def test_accepted_fixture_exports_one_complete_multiturn_episode(self):
        self.accepted_fixture()
        output = self.root / "release"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rollout.export_release(self.root, self.plan, output), 0)
        rows = [json.loads(line) for line in (output / "sft/train.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["messages"]), 6)
        self.assertEqual(rows[0]["supervised_assistant_turns"], 3)
        self.assertEqual(rollout.read(output / "manifest.json")["status"], "complete")

    def test_accepted_review_cannot_bypass_an_anchor_that_changed_before_export(self):
        _, _, anchor = self.accepted_fixture()
        baseline = (self.root / anchor["nop_result"]).parent / "artifacts/app/LAPS" / SOURCE
        baseline.write_text("tampered defective baseline after review")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            rollout.export_release(self.root, self.plan, self.root / "release")


class ResumeTests(FixtureCase):
    def state(self, status="started", with_trial=False):
        folder = self.root / "attempts/fixture/attempt-001"
        folder.mkdir(parents=True)
        state = {
            "status": status,
            "pid": 987654321,
            "agent": "terminus-2",
            "plan_sha256": rollout.digest(self.root / "plan.json"),
            "task_files": rollout.files(self.root / self.task["path"]),
        }
        path = self.trial("attempts/fixture/attempt-001/jobs/job/trial") if with_trial else None
        if path:
            state.update(result=str(path.relative_to(self.root)), result_sha256=rollout.digest(path))
        rollout.write(folder / "execution.json", state)
        return folder, path

    def test_live_original_child_blocks_duplicate_without_spawning(self):
        folder, _ = self.state()
        ps = subprocess.CompletedProcess(
            [], 0, stdout="python harbor run --jobs-dir " + str(folder.resolve()), stderr=""
        )
        with (
            mock.patch.object(rollout.subprocess, "run", return_value=ps),
            mock.patch.object(rollout.subprocess, "Popen") as spawn,
        ):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                rollout.execute_job(self.root, self.plan, self.task, folder, "terminus-2")
            spawn.assert_not_called()

    def test_dead_started_child_without_trial_is_preserved_without_new_api_call(self):
        folder, _ = self.state()
        with (
            mock.patch.object(rollout, "child_alive", return_value=False),
            mock.patch.object(rollout.subprocess, "Popen") as spawn,
        ):
            self.assertIsNone(rollout.execute_job(self.root, self.plan, self.task, folder, "terminus-2"))
            self.assertEqual(rollout.read(folder / "execution.json")["status"], "interrupted_preserved")
            spawn.assert_not_called()

    def test_completed_result_reconciles_started_receipt_without_new_rollout(self):
        folder, path = self.state(with_trial=True)
        with (
            mock.patch.object(rollout, "child_alive", return_value=False),
            mock.patch.object(rollout.subprocess, "Popen") as spawn,
        ):
            self.assertEqual(rollout.execute_job(self.root, self.plan, self.task, folder, "terminus-2"), path)
            self.assertEqual(rollout.read(folder / "execution.json")["status"], "finished")
            spawn.assert_not_called()

    def test_finished_receipt_cannot_reuse_a_tampered_result(self):
        folder, path = self.state(status="finished", with_trial=True)
        path.write_text(path.read_text() + " ")
        with (
            mock.patch.object(rollout, "child_alive", return_value=False),
            mock.patch.object(rollout.subprocess, "Popen") as spawn,
        ):
            with self.assertRaisesRegex(ValueError, "Stored raw result changed"):
                rollout.execute_job(self.root, self.plan, self.task, folder, "terminus-2")
            spawn.assert_not_called()

    def test_finished_receipt_cannot_reuse_result_for_changed_task_inputs(self):
        folder, _ = self.state(status="finished", with_trial=True)
        (self.root / self.task["path"] / "instruction.md").write_text("changed inputs")
        with (
            mock.patch.object(rollout, "child_alive", return_value=False),
            mock.patch.object(rollout.subprocess, "Popen") as spawn,
        ):
            with self.assertRaisesRegex(ValueError, "(?i)(task|inputs).*(changed|differ|mismatch)"):
                rollout.execute_job(self.root, self.plan, self.task, folder, "terminus-2")
            spawn.assert_not_called()


class FakeTokenizer:
    """Character-tokenized prefix-stable template; no model files or downloads."""

    def __init__(self, drop_thinking=False):
        self.drop_thinking = drop_thinking
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        self.calls.append(kwargs)
        text = ""
        for message in messages:
            if message["role"] == "assistant":
                thought = "" if self.drop_thinking else message.get("reasoning_content", "")
                text += "<assistant>\n<think>\n" + thought + "\n</think>\n" + message["content"] + "<end>\n"
            else:
                text += "<" + message["role"] + ">\n" + message["content"] + "<end>\n"
        if add_generation_prompt:
            text += "<assistant>\n<think>\n"
        return text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return argparse.Namespace(input_ids=[ord(char) for char in text])


class TokenizerContractTests(unittest.TestCase):
    def row(self):
        result, atif = dialogue(["Native initial reasoning", None, "Native final reasoning"])
        messages, _ = rollout.align_episode(result, atif)
        return {"episode_id": "fixture--attempt-001", "messages": messages}

    def test_masks_only_reviewed_assistants_and_keeps_native_reasoning(self):
        row, tokenizer = self.row(), FakeTokenizer()
        encoded = tokenize_qwen.encode_episode(row, tokenizer, 10000)
        self.assertEqual([span["message_index"] for span in encoded["supervised_assistant_spans"]], [1, 5])
        supervised_positions = set()
        for span in encoded["supervised_assistant_spans"]:
            supervised_positions.update(range(span["start"], span["end"]))
        for index, label in enumerate(encoded["labels"]):
            self.assertEqual(label, encoded["input_ids"][index] if index in supervised_positions else -100)
        text = "".join(chr(token) for token in encoded["input_ids"])
        self.assertIn("Native initial reasoning", text)
        self.assertIn("Native final reasoning", text)
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))
        self.assertTrue(all(call == {"enable_thinking": True, "preserve_thinking": True} for call in tokenizer.calls))
        self.assertEqual(row["messages"][3]["reasoning_content"], "")
        self.assertFalse(row["messages"][3]["loss"])

    def test_complete_episode_over_length_is_rejected_without_truncation(self):
        row = self.row()
        with self.assertRaisesRegex(ValueError, "not truncated"):
            tokenize_qwen.encode_episode(row, FakeTokenizer(), 1)
        self.assertEqual(len(row["messages"]), 6)

    def test_template_dropping_native_reasoning_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "drops native reasoning"):
            tokenize_qwen.encode_episode(self.row(), FakeTokenizer(drop_thinking=True), 10000)

    def test_template_cannot_drop_reasoning_from_context_only_assistant(self):
        class DropsMaskedThinking(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return super().apply_chat_template(messages, **kwargs).replace("Masked native reasoning", "")

        row = self.row()
        row["messages"][3]["reasoning_content"] = "Masked native reasoning"
        with self.assertRaisesRegex(ValueError, "drops native reasoning"):
            tokenize_qwen.encode_episode(row, DropsMaskedThinking(), 10000)

    def test_missing_native_reasoning_cannot_be_enabled_for_loss(self):
        row = self.row()
        row["messages"][3]["loss"] = True
        with self.assertRaisesRegex(ValueError, "lacks native reasoning"):
            tokenize_qwen.encode_episode(row, FakeTokenizer(), 10000)


class ReleaseIntegrityTests(FixtureCase):
    def release(self):
        root = self.root / "release"
        path = root / "sft/train.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"episode_id": "synthetic"}) + "\n")
        rollout.write(
            root / "manifest.json",
            {"schema": rollout.SCHEMA, "status": "complete", "files": {"sft/train.jsonl": rollout.digest(path)}},
        )
        return root

    def test_consumed_sft_must_be_explicitly_hashed(self):
        root = self.release()
        self.assertEqual(tokenize_qwen.verified_release(root), [{"episode_id": "synthetic"}])
        manifest = rollout.read(root / "manifest.json")
        manifest["files"] = {}
        rollout.write(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "does not authenticate"):
            tokenize_qwen.verified_release(root)

    def test_modified_sft_is_rejected_before_loading_tokenizer(self):
        root = self.release()
        (root / "sft/train.jsonl").write_text('{"episode_id":"modified"}\n')
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            tokenize_qwen.verified_release(root)

    def test_release_evidence_cannot_escape_root(self):
        root = self.release()
        outside = self.root / "unrelated.txt"
        outside.write_text("unrelated")
        manifest = rollout.read(root / "manifest.json")
        manifest["files"]["../unrelated.txt"] = rollout.digest(outside)
        rollout.write(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            tokenize_qwen.verified_release(root)

    def test_changed_alignment_module_invalidates_the_run(self):
        self.plan["runtime"]["code_sha256"]["alignment.py"] = "different-review-policy"
        rollout.write(self.root / "plan.json", self.plan)
        (self.root / "plan.sha256").write_text(rollout.digest(self.root / "plan.json"))
        with self.assertRaisesRegex(ValueError, "module code changed"):
            rollout.load_plan(self.root)


if __name__ == "__main__":
    unittest.main()
