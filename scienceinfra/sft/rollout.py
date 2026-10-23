#!/usr/bin/env python3
"""Collect and export reviewed LAPS SFT episodes using Harbor and a remote teacher."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scienceinfra.sft.alignment import align_episode

SCIENCEINFRA_COMMIT = "49bd221bb18077a349e7f860569c3c5f5a4de64f"
HARBOR_COMMIT = "3c82380859d187957cfd5cd64802b076d9779550"
SCHEMA = "scienceinfra_sft_v1"
SOURCE_SUFFIXES = {".f90", ".f", ".for", ".c", ".cc", ".cpp", ".h", ".hpp", ".inc"}
API_BLOCKER = re.compile(
    r"AuthenticationError|AgentAuthenticationError|PermissionDeniedError|"
    r"ModelNotFoundError|RateLimitError|ApiUsageLimitError|insufficient_quota|invalid_api_key|model_not_found",
    re.I,
)


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def files(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed in evidence tree: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def summarize(values):
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def safe_id(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:70]
    return slug + "-" + hashlib.sha256(name.encode()).hexdigest()[:10]


@contextlib.contextmanager
def lock(root):
    with (root / "runner.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another command owns this run. Do not start two schedulers.") from None
        yield


def load_plan(root):
    path = root / "plan.json"
    if digest(path) != (root / "plan.sha256").read_text().strip():
        raise ValueError("Run plan changed. Use a new run directory for new settings.")
    plan = read(path)
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported run schema")
    if plan.get("runtime", {}).get("starter_sha256") != digest(__file__):
        raise ValueError("Starter code changed since init; use the original kit or a new run")
    if plan.get("runtime", {}).get("code_sha256") != code_fingerprints():
        raise ValueError("SFT module code changed since init; use the original revision or a new run")
    for task in plan["tasks"]:
        if files(root / task["path"]) != task["task_files"]:
            raise ValueError(f"Staged task changed: {task['task_name']}")
    return plan


def code_fingerprints():
    """Bind review policy and encoding code as well as the collector to a run."""
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def runtime_record():
    versions = {}
    for package in ["harbor", "scienceinfra", "pandas", "pyarrow", "litellm"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "versions": versions,
        "platform": sys.platform,
        "starter_sha256": digest(__file__),
        "code_sha256": code_fingerprints(),
    }


def no_network(text):
    """Override runtime policy in the staged copy, preserving scientific fields."""
    import toml

    original = tomllib.loads(text)
    changed = tomllib.loads(text)
    for section in ["environment", "agent"]:
        if section not in changed:
            raise ValueError(f"Task has no [{section}] section")
        changed[section]["network_mode"] = "no-network"
        changed[section]["allowed_hosts"] = []
    text = toml.dumps(changed)
    changed = tomllib.loads(text)
    for section in ["environment", "agent"]:
        for key in ["network_mode", "allowed_hosts"]:
            original.get(section, {}).pop(key, None)
            changed.get(section, {}).pop(key, None)
    if original != changed:
        raise ValueError("Network staging changed non-network task fields")
    return text


def init_run(args):
    import pandas as pd

    root = args.run_dir.resolve()
    if root.exists():
        raise ValueError("init requires a NEW run directory")
    if args.attempts < 1 or args.concurrency < 1 or args.limit < 1:
        raise ValueError("attempts, concurrency and limit must be positive")
    parsed = urllib.parse.urlparse(args.api_base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("API base must be HTTPS without credentials")
    df = pd.read_parquet(args.dataset)
    selected = []
    for row in df.to_dict("records"):
        info = row["extra_info"]
        if int(info.get("gpus") or 0) > 0:
            continue
        if info.get("reward_key") != "reward_repair":
            continue
        selected.append((row, info))
        if len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError("No selected CPU repair/implementation tasks")
    root.mkdir(parents=True)
    tasks = []
    for row, info in selected:
        source = Path(info["task_path"]).resolve()
        task_id = safe_id(row["task_name"])
        target = root / "tasks" / task_id / "task"
        recipe = source / "solution/solve.sh"
        if not recipe.is_file() or "/app/LAPS" not in recipe.read_text():
            raise ValueError("This starter's source replay adapter requires compiled LAPS tasks")
        original_files = files(source)
        shutil.copytree(source, target)
        manifest = target / "task.toml"
        manifest.write_text(no_network(manifest.read_text()))
        config = tomllib.loads(manifest.read_text())
        if config.get("task", {}).get("name") != row["task_name"]:
            raise ValueError("Parquet task_name does not match the compiled task manifest")
        checks = config.get("metadata", {}).get("taxonomy", {}).get("affected_checks")
        if not checks:
            raise ValueError(f"Task does not declare affected_checks: {source}")
        tasks.append(
            {
                "id": task_id,
                "task_name": row["task_name"],
                "path": str(target.relative_to(root)),
                "origin_path": str(source),
                "origin_task_files": original_files,
                "task_files": files(target),
                "hint": str(info.get("hint") or ""),
                "hint_level": info.get("hint_level"),
                "checks": list(checks),
                "estimated_floor": float(row.get("floor", 0)),
                "source_provenance": config.get("metadata", {}).get("provenance", {}),
                "network_adaptation": "staged environment+agent runtime no-network",
            }
        )
    plan = {
        "schema": SCHEMA,
        "created_at": now(),
        "dataset_origin": str(args.dataset.resolve()),
        "dataset_sha256": digest(args.dataset),
        "teacher_model": args.model,
        "api_base": args.api_base.rstrip("/"),
        "teacher_revision": "unattested",
        "attempts_per_task": args.attempts,
        "concurrency": args.concurrency,
        "max_turns": args.max_turns,
        "max_output_tokens": args.max_output_tokens,
        "context_tokens": args.context_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "source_root": "/app/LAPS",
        "source_replay_adapter": "laps-pinned-solve-v1",
        "tasks": tasks,
        "runtime": runtime_record(),
        "expected_upstream": {"scienceinfra": SCIENCEINFRA_COMMIT, "harbor": HARBOR_COMMIT},
    }
    write(root / "plan.json", plan)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n")
    print(
        json.dumps(
            {"run_dir": str(root), "tasks": len(tasks), "planned_teacher_attempts": len(tasks) * args.attempts},
            ensure_ascii=False,
        )
    )


def need_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment; never put it in the run plan")
    return key


def preflight(root, plan):
    key = need_key()
    payload = {
        "model": plan["teacher_model"],
        "messages": [{"role": "user", "content": "Compute 17 times 19. Return the number."}],
        "max_tokens": 1024,
        "temperature": 0.6,
    }
    request = urllib.request.Request(
        plan["api_base"] + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    folder = root / "preflight" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().replace(key.encode(), b"[REDACTED]")
        (folder / "http_error.txt").write_bytes(body)
        write(folder / "result.json", {"status": "failed", "http_status": error.code, "at": now()})
        raise ValueError(f"Provider preflight HTTP {error.code}; see {folder}") from None
    raw = raw.replace(key.encode(), b"[REDACTED]")
    (folder / "response.json").write_bytes(raw)
    data = json.loads(raw)
    message = data["choices"][0]["message"]
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip() or not message.get("content"):
        raise ValueError(f"API did not return native reasoning plus content; raw response saved at {folder}")
    receipt = {
        "status": "passed",
        "at": now(),
        "model": plan["teacher_model"],
        "returned_model": data.get("model"),
        "api_base": plan["api_base"],
        "response": str((folder / "response.json").relative_to(root)),
        "response_sha256": digest(folder / "response.json"),
        "reasoning_characters": len(reasoning),
        "usage": data.get("usage"),
    }
    write(folder / "result.json", receipt)
    write(root / "provider_preflight.json", receipt)
    print(json.dumps({"status": "passed", "receipt": str(folder / "result.json")}, ensure_ascii=False))


def verify_preflight(root, plan):
    receipt = read(root / "provider_preflight.json")
    if (
        receipt.get("status") != "passed"
        or receipt["model"] != plan["teacher_model"]
        or receipt["api_base"] != plan["api_base"]
        or digest(root / receipt["response"]) != receipt["response_sha256"]
    ):
        raise ValueError("Missing or changed matching provider preflight")


def trial_result(folder):
    candidates = []
    for path in (folder / "jobs").rglob("result.json"):
        try:
            data = read(path)
        except (ValueError, OSError):
            continue
        if "task_checksum" in data and "task_name" in data and data.get("finished_at"):
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"More than one trial in an independent slot: {folder}")
    return candidates[0] if candidates else None


def child_alive(state, folder):
    pid = state.get("pid")
    if not isinstance(pid, int):
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
    return result.returncode == 0 and "harbor" in result.stdout and str(folder.resolve()) in result.stdout


def job_command(root, plan, task, folder, agent, task_path=None):
    path = task_path or root / task["path"]
    command = [
        "harbor",
        "run",
        "-p",
        str(path.resolve()),
        "-a",
        agent,
        "-k",
        "1",
        "-n",
        "1",
        "--max-retries",
        "0",
        "--yes",
        "--jobs-dir",
        str((folder / "jobs").resolve()),
        "--job-name",
        "job",
        "--artifact",
        plan["source_root"],
    ]
    if agent == "terminus-2":
        options = {
            "api_base": plan["api_base"],
            "parser_name": "json",
            "max_turns": plan["max_turns"],
            "temperature": plan["temperature"],
            "enable_summarize": False,
            "store_all_messages": True,
            "record_terminal_session": False,
            "tmux_pane_width": 160,
            "tmux_pane_height": 160,
            "trajectory_config": {"raw_content": True},
            "llm_kwargs": {"timeout": 180, "max_retries": 0},
            "llm_call_kwargs": {"max_tokens": plan["max_output_tokens"], "top_p": plan["top_p"]},
            "model_info": {
                "max_input_tokens": plan["context_tokens"] - plan["max_output_tokens"],
                "max_output_tokens": plan["max_output_tokens"],
                "max_tokens": plan["context_tokens"],
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
        }
        command += ["-m", "openai/" + plan["teacher_model"]]
        for key, value in options.items():
            command += ["--ak", key + "=" + (value if isinstance(value, str) else json.dumps(value))]
        if task["hint"]:
            command += ["--extra-instruction", task["hint"]]
    return command


def execute_job(root, plan, task, folder, agent, task_path=None):
    """One slot never silently becomes a new paid attempt after interruption."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "execution.json"
    if receipt_path.exists():
        state = read(receipt_path)
        if state.get("task_files") != files(task_path or root / task["path"]):
            raise ValueError("Task/replay inputs differ from the recorded execution")
        if child_alive(state, folder):
            raise RuntimeError(f"Original Harbor child is still active: {folder}")
        result = trial_result(folder)
        if state["status"] == "started":
            state.update(status="finished" if result else "interrupted_preserved", reconciled_at=now())
            if result:
                state.update(result=str(result.relative_to(root)), result_sha256=digest(result))
            write(receipt_path, state)
        if state.get("result"):
            path = root / state["result"]
            if digest(path) != state["result_sha256"]:
                raise ValueError(f"Stored raw result changed: {path}")
            return path
        return None
    command = job_command(root, plan, task, folder, agent, task_path)
    key = need_key() if agent == "terminus-2" else os.environ.get("OPENAI_API_KEY", "")
    state = {
        "status": "started",
        "started_at": now(),
        "agent": agent,
        "command": command,
        "plan_sha256": digest(root / "plan.json"),
        "task_files": files(task_path or root / task["path"]),
    }
    write(receipt_path, state)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        state["pid"] = process.pid
        write(receipt_path, state)
        try:
            with (folder / "collector.log").open("w") as log:
                for line in process.stdout:
                    log.write(line.replace(key, "[REDACTED]") if key else line)
                    log.flush()
            returncode = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    except BaseException as error:
        state.update(status="interrupted_preserved", finished_at=now(), error=type(error).__name__)
        write(receipt_path, state)
        raise
    result = trial_result(folder)
    state.update(
        status="finished",
        returncode=returncode,
        finished_at=now(),
        wall_seconds=round(time.monotonic() - started, 3),
        log_sha256=digest(folder / "collector.log"),
    )
    if result:
        state.update(result=str(result.relative_to(root)), result_sha256=digest(result))
    write(receipt_path, state)
    return result


def reward_record(result_path):
    result = read(result_path)
    if result.get("exception_info") is not None:
        raise ValueError("runtime_exception")
    path = result_path.parent / "verifier/reward.json"
    payload = read(path)
    rewards = payload.get("rewards", payload)
    if not isinstance(rewards, dict) or rewards != (result.get("verifier_result") or {}).get("rewards"):
        raise ValueError("reward_result_mismatch")
    for key in ["reward", "floor", "reward_repair"]:
        value = rewards.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("missing_numeric_" + key)
    floor = rewards["floor"]
    if not 0 <= floor < 1:
        raise ValueError("invalid_measured_floor")
    derived = round(max(0, rewards["reward"] - floor) / max(1e-6, 1 - floor), 6)
    if abs(derived - rewards["reward_repair"]) > 1e-6:
        raise ValueError("repair_score_inconsistent")
    if rewards.get("grader_error") or rewards.get("grader_exception"):
        raise ValueError("grader_error")
    return result, rewards


def scientific_pass(result_path, task, floor=None):
    result, rewards = reward_record(result_path)
    if result.get("task_name") != task["task_name"]:
        raise ValueError("trial_task_name_mismatch")
    if floor is not None and abs(rewards["floor"] - floor) > 1e-6:
        raise ValueError("measured_floor_drift")
    if any(rewards.get(k) != 1 for k in ["reward", "reward_repair", "equivalence_pass"]):
        raise ValueError("scientific_score_below_one")
    for check in task["checks"]:
        if rewards.get("check_" + check.replace("-", "_")) != 1:
            raise ValueError("required_check_failed:" + check)
    report = read(result_path.parent / "verifier/grading_report.json")
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("unsupported_grading_report")
    for check in task["checks"]:
        if checks.get(check, {}).get("passed") is not True:
            raise ValueError("grading_report_failed:" + check)
    return result, rewards


def verify_anchor(root, task):
    folder = root / "anchors" / task["id"]
    receipt = read(folder / "anchor.json")
    if receipt.get("status") != "passed":
        raise ValueError("oracle/nop anchor not passed")
    for rel, expected in receipt.get("evidence_sha256", {}).items():
        if digest(root / rel) != expected:
            raise ValueError("Anchor evidence changed: " + rel)
    for kind in ["oracle", "nop"]:
        path = root / receipt[kind + "_result"]
        if digest(path) != receipt[kind + "_result_sha256"]:
            raise ValueError("Anchor raw result changed")
    scientific_pass(root / receipt["oracle_result"], task, receipt["floor"])
    nop_result, nop = reward_record(root / receipt["nop_result"])
    if nop_result.get("task_name") != task["task_name"]:
        raise ValueError("nop_task_name_mismatch")
    if abs(nop["floor"] - receipt["floor"]) > 1e-6 or nop["reward_repair"] != 0:
        raise ValueError("Nop anchor changed")
    baseline = (root / receipt["nop_result"]).parent / "artifacts/app/LAPS"
    if source_files(baseline) != receipt["baseline_sources"]:
        raise ValueError("Captured defective baseline source changed")
    return receipt


def anchors(root, plan):
    for task in plan["tasks"]:
        parent = root / "anchors" / task["id"]
        if (parent / "anchor.json").exists():
            verify_anchor(root, task)
            print("anchor reused:", task["task_name"], flush=True)
            continue
        oracle = execute_job(root, plan, task, parent / "oracle", "oracle")
        nop = execute_job(root, plan, task, parent / "nop", "nop")
        if oracle is None or nop is None:
            raise ValueError(f"Anchor lacks completed trial: {parent}; inspect logs and use a new run after fixing it")
        _, full = scientific_pass(oracle, task)
        nop_result, zero = reward_record(nop)
        if nop_result.get("task_name") != task["task_name"]:
            raise ValueError("nop_task_name_mismatch")
        if zero["reward_repair"] != 0 or abs(zero["reward"] - zero["floor"]) > 1e-6:
            raise ValueError("Nop does not match the measured defective baseline")
        if abs(full["floor"] - zero["floor"]) > 1e-6:
            raise ValueError("Oracle/nop floor mismatch")
        evidence = [
            oracle,
            nop,
            oracle.parent / "verifier/reward.json",
            oracle.parent / "verifier/grading_report.json",
            nop.parent / "verifier/reward.json",
        ]
        write(
            parent / "anchor.json",
            {
                "status": "passed",
                "at": now(),
                "floor": zero["floor"],
                "oracle_result": str(oracle.relative_to(root)),
                "oracle_result_sha256": digest(oracle),
                "nop_result": str(nop.relative_to(root)),
                "nop_result_sha256": digest(nop),
                "baseline_sources": source_files(nop.parent / "artifacts/app/LAPS"),
                "evidence_sha256": {str(p.relative_to(root)): digest(p) for p in evidence},
            },
        )
        print("anchor passed:", task["task_name"], flush=True)


def collect(root, plan):
    verify_preflight(root, plan)
    need_key()
    for task in plan["tasks"]:
        verify_anchor(root, task)
    if (root / "api_blocker.json").exists():
        raise ValueError("api_blocker.json exists. Diagnose it, preserve it under history, then retry this command.")
    stop = threading.Event()

    def one(task, slot):
        if stop.is_set():
            return
        folder = root / "attempts" / task["id"] / f"attempt-{slot:03d}"
        result = execute_job(root, plan, task, folder, "terminus-2")
        evidence = (folder / "collector.log").read_text(errors="replace") if (folder / "collector.log").exists() else ""
        if result:
            evidence += json.dumps(read(result).get("exception_info"))
        if API_BLOCKER.search(evidence):
            stop.set()
            # This contains paths and a classification only, never provider credentials.
            write(
                root / "api_blocker.json",
                {
                    "at": now(),
                    "attempt": str(folder.relative_to(root)),
                    "reason": "authentication/permission/model/quota error",
                },
            )
        print(f"finished {task['task_name']} slot={slot} has_trial={result is not None}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["concurrency"]) as pool:
        futures = [
            pool.submit(one, task, slot) for task in plan["tasks"] for slot in range(1, plan["attempts_per_task"] + 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException:
                stop.set()
                for pending in futures:
                    pending.cancel()
                raise
    if stop.is_set():
        raise ValueError("API blocker detected; unstarted slots were not dispatched")


def source_files(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError("missing_captured_source")
    result = {rel: value for rel, value in files(path).items() if Path(rel).suffix.lower() in SOURCE_SUFFIXES}
    if not result:
        raise ValueError("no_scientific_source_files")
    return result


def build_replay(root, plan, task, result_path, folder):
    """LAPS adapter: no official fix; only captured scientific source is injected."""
    anchor = verify_anchor(root, task)
    baseline = (root / anchor["nop_result"]).parent / "artifacts/app/LAPS"
    candidate = result_path.parent / "artifacts/app/LAPS"
    before, after = source_files(baseline), source_files(candidate)
    if before.keys() != after.keys():
        raise ValueError("source_file_set_changed_not_supported")
    changed = [rel for rel in before if before[rel] != after[rel]]
    if not changed:
        raise ValueError("no_scientific_source_change")
    replay = folder / "replay-task"
    expected = {
        "baseline_sources": before,
        "candidate_sources": after,
        "changed_sources": changed,
        "adapter": plan["source_replay_adapter"],
    }
    receipt = folder / "source_changes.json"
    if replay.exists():
        if read(receipt) != expected:
            raise ValueError("Source replay inputs changed")
        if files(replay) != read(folder / "replay_task_files.json"):
            raise ValueError("Prepared replay task changed")
        return replay
    shutil.copytree(root / task["path"], replay)
    solution = replay / "solution"
    script = solution / "solve.sh"
    text = script.read_text()
    needle = 'python3 "$HERE/apply_defect.py" "$HERE/defect_fix.json" /app/LAPS'
    if text.count(needle) != 1:
        raise ValueError("unsupported_laps_solution_recipe")
    script.write_text(text.replace(needle, 'python3 "$HERE/replay_source.py"', 1))
    for rel in after:
        target = solution / "sampled_source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / rel, target)
    write(solution / "source_manifest.json", expected)
    (solution / "replay_source.py").write_text(
        "import hashlib,json,shutil\nfrom pathlib import Path\n"
        'here=Path(__file__).resolve().parent; root=Path("/app/LAPS")\n'
        'm=json.loads((here/"source_manifest.json").read_text())\n'
        'for rel,expected in m["baseline_sources"].items():\n'
        " p=root/rel\n"
        ' if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError("baseline source drift: "+rel)\n'
        ' q=here/"sampled_source"/rel\n'
        ' if hashlib.sha256(q.read_bytes()).hexdigest()!=m["candidate_sources"][rel]:\n'
        '  raise ValueError("sampled source changed")\n'
        " shutil.copyfile(q,p)\n"
        'for p in root.rglob("*"):\n'
        ' if p.is_file() and p.suffix.lower() in {".o",".mod",".smod",".exe",".a",".so"}: p.unlink()\n'
    )
    for name in ["defect_fix.json", "apply_defect.py"]:
        (solution / name).unlink(missing_ok=True)
    write(receipt, expected)
    write(folder / "replay_task_files.json", files(replay))
    return replay


def attempt_folders(root, plan):
    for task in plan["tasks"]:
        for slot in range(1, plan["attempts_per_task"] + 1):
            yield task, slot, root / "attempts" / task["id"] / f"attempt-{slot:03d}"


def review_one(root, plan, task, slot, folder):
    receipt_path = folder / "review.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for rel, expected in receipt.get("evidence_sha256", {}).items():
            if digest(root / rel) != expected:
                raise ValueError("Reviewed evidence changed: " + rel)
        return receipt
    if not (folder / "execution.json").exists():
        return {"status": "not_started"}
    execution = read(folder / "execution.json")
    if execution["status"] == "started":
        raise ValueError(f"Run rollout again to reconcile this started slot: {folder}")
    record = {
        "status": "excluded",
        "at": now(),
        "task_id": task["id"],
        "slot": slot,
        "reviewer": "automatic scientific+alignment+fresh-source-replay; not an independent expert",
        "evidence_sha256": {str((folder / "execution.json").relative_to(root)): digest(folder / "execution.json")},
    }
    try:
        if not execution.get("result"):
            raise ValueError("no_completed_harbor_trial")
        path = root / execution["result"]
        if digest(path) != execution["result_sha256"]:
            raise ValueError("attempt_result_changed")
        required = [
            path,
            path.parent / "agent/trajectory.json",
            path.parent / "verifier/reward.json",
            path.parent / "verifier/grading_report.json",
        ]
        record["evidence_sha256"].update({str(p.relative_to(root)): digest(p) for p in required})
        anchor = verify_anchor(root, task)
        anchor_path = root / "anchors" / task["id"] / "anchor.json"
        record["evidence_sha256"][str(anchor_path.relative_to(root))] = digest(anchor_path)
        result, _ = scientific_pass(path, task, anchor["floor"])
        atif_path = path.parent / "agent/trajectory.json"
        if list(atif_path.parent.glob("trajectory*.json")) != [atif_path]:
            raise ValueError("unsupported_multiple_trajectories")
        messages, masks = align_episode(result, read(atif_path))
        replay = build_replay(root, plan, task, path, folder)
        record["stage"] = "source_replay"
        replay_result = execute_job(root, plan, task, folder / "source-replay", "oracle", replay)
        if replay_result is None:
            raise ValueError("source_replay_has_no_completed_trial")
        change_path = folder / "source_changes.json"
        record["evidence_sha256"].update(
            {
                str(p.relative_to(root)): digest(p)
                for p in [
                    change_path,
                    replay_result,
                    replay_result.parent / "verifier/reward.json",
                    replay_result.parent / "verifier/grading_report.json",
                ]
            }
        )
        scientific_pass(replay_result, task, anchor["floor"])
        record.update(
            status="accepted",
            reason=None,
            assistant_turns=len(masks),
            source_replay_result=str(replay_result.relative_to(root)),
            supervised_assistant_turns=sum(m["loss"] for m in masks),
            native_reasoning_turns=sum(bool(m.get("reasoning_content")) for m in messages if m["role"] == "assistant"),
            turn_review=masks,
            source_replay_task_files=files(replay),
            source_artifacts=source_files(path.parent / "artifacts/app/LAPS"),
        )
    except (ValueError, KeyError, OSError) as error:
        record["reason"] = str(error)
    write(receipt_path, record)
    return record


def retry_failed_replay(root, folder):
    """Explicit CPU-only retry; preserve the failed review and raw replay job."""
    path = folder / "review.json"
    if not path.exists():
        return
    review = read(path)
    if review.get("status") != "excluded" or review.get("stage") != "source_replay":
        return
    execution = folder / "source-replay/execution.json"
    if execution.exists() and child_alive(read(execution), folder / "source-replay"):
        raise ValueError("Previous source replay is still active")
    target = folder / "review-history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True, exist_ok=False)
    receipt = {
        "at": now(),
        "action": "explicit_cpu_only_replay_retry",
        "teacher_calls": 0,
        "original_review_sha256": digest(path),
        "archives": {},
    }
    for name in ["source-replay", "review.json"]:
        origin = folder / name
        if origin.exists():
            receipt["archives"][name] = files(origin) if origin.is_dir() else digest(origin)
            shutil.move(str(origin), str(target / name))
    write(target / "archive.json", receipt)


def reviews(root, plan, retry_replay=False):
    for task, slot, folder in attempt_folders(root, plan):
        if retry_replay:
            retry_failed_replay(root, folder)
        record = review_one(root, plan, task, slot, folder)
        print(task["task_name"], slot, record["status"], record.get("reason"), flush=True)


def export_release(root, plan, output):
    if output.exists():
        raise ValueError("Export output must be a NEW directory")
    episodes, attempts, receipts = [], [], []
    for task, slot, folder in attempt_folders(root, plan):
        state = read(folder / "execution.json") if (folder / "execution.json").exists() else {}
        row = {
            "task_name": task["task_name"],
            "slot": slot,
            "status": state.get("status", "not_started"),
            "scientific_success": False,
            "review_status": "not_reviewed",
            "reason": None,
        }
        if state.get("result"):
            path = root / state["result"]
            if digest(path) != state["result_sha256"]:
                raise ValueError("Raw result changed")
            data = read(path)
            agent = data.get("agent_result") or {}
            raw_messages = agent.get("metadata", {}).get("all_messages") or []
            row.update(
                api_input_tokens=agent.get("n_input_tokens"),
                api_output_tokens=agent.get("n_output_tokens"),
                exception_info=data.get("exception_info"),
                raw_result=str(path.relative_to(root)),
                raw_user_turns=sum(m.get("role") == "user" for m in raw_messages),
                raw_assistant_turns=sum(m.get("role") == "assistant" for m in raw_messages),
            )
            try:
                scientific_pass(path, task, verify_anchor(root, task)["floor"])
                row["scientific_success"] = True
            except (ValueError, KeyError, OSError):
                pass
        if (folder / "review.json").exists():
            review = review_one(root, plan, task, slot, folder)
            row.update(review_status=review["status"], reason=review.get("reason"))
            if review["status"] == "accepted":
                if not row["scientific_success"]:
                    raise ValueError("Accepted attempt no longer passes its scientific/anchor checks")
                anchor = verify_anchor(root, task)
                scientific_pass(root / review["source_replay_result"], task, anchor["floor"])
                path = root / state["result"]
                messages, masks = align_episode(read(path), read(path.parent / "agent/trajectory.json"))
                if masks != review["turn_review"]:
                    raise ValueError("Review policy drift")
                if source_files(path.parent / "artifacts/app/LAPS") != review["source_artifacts"]:
                    raise ValueError("Model source artifact changed")
                if files(folder / "replay-task") != review["source_replay_task_files"]:
                    raise ValueError("Replay task changed")
                episode = {
                    "schema": SCHEMA,
                    "sft_kind": "complete_multiturn_teacher_episode",
                    "episode_id": task["id"] + f"--attempt-{slot:03d}",
                    "task_name": task["task_name"],
                    "teacher_model": plan["teacher_model"],
                    "teacher_revision": "unattested",
                    "source_provenance": task.get("source_provenance", {}),
                    "hint_level": task["hint_level"],
                    "messages": messages,
                    "tool_protocol": "terminus-2-json-raw-content",
                    "assistant_turns": len(masks),
                    "supervised_assistant_turns": sum(m["loss"] for m in masks),
                    "native_reasoning_turns": review["native_reasoning_turns"],
                    "source_attempt": str(folder.relative_to(root)),
                    "review_sha256": digest(folder / "review.json"),
                }
                episodes.append(episode)
                receipts.append(review)
        attempts.append(row)
    output.mkdir(parents=True)

    def jsonl(path, rows):
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    jsonl(output / "sft/train.jsonl", episodes)
    jsonl(output / "attempts.jsonl", attempts)
    jsonl(output / "tasks.jsonl", plan["tasks"])
    write(output / "evidence.json", receipts)
    finished = sum(a["status"] in {"finished", "interrupted_preserved"} for a in attempts)
    stats = {
        "planned_tasks": len(plan["tasks"]),
        "planned_attempts": len(attempts),
        "ended_attempts": finished,
        "scientific_successes": sum(a["scientific_success"] for a in attempts),
        "sft_episodes": len(episodes),
        "sft_distinct_tasks": len({e["task_name"] for e in episodes}),
        "scientific_success_rate_ended": sum(a["scientific_success"] for a in attempts) / finished
        if finished
        else None,
        "sft_admission_rate_ended": len(episodes) / finished if finished else None,
        "attempt_status_counts": dict(Counter(a["status"] for a in attempts)),
        "review_status_counts": dict(Counter(a["review_status"] for a in attempts)),
        "exclusion_reasons": dict(Counter(a["reason"] for a in attempts if a["reason"])),
        "raw_attempt_user_rounds": summarize([a["raw_user_turns"] for a in attempts if "raw_user_turns" in a]),
        "sft_user_rounds": summarize([e["assistant_turns"] for e in episodes]),
        "sft_native_reasoning_turns": summarize([e["native_reasoning_turns"] for e in episodes]),
        "sft_supervised_assistant_turns": summarize([e["supervised_assistant_turns"] for e in episodes]),
        "api_input_tokens_completed": sum(a.get("api_input_tokens") or 0 for a in attempts),
        "api_output_tokens_completed": sum(a.get("api_output_tokens") or 0 for a in attempts),
        "notes": [
            "API tokens are cumulative calls, not target-tokenized SFT sequence lengths.",
            "user round count includes the task and terminal feedback, not human turns.",
            "No target tokenizer has been applied; JSONL loss fields require a compatible trainer adapter.",
            "Only accepted, aligned, completed native-reasoning episodes with fresh LAPS source replay are exported.",
        ],
    }
    write(output / "statistics.json", stats)
    status = (
        "complete" if finished == len(attempts) and episodes else "partial" if episodes else "no_admissible_episodes"
    )
    write(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "status": status,
            "at": now(),
            "model_training_performed": False,
            "plan_sha256": digest(root / "plan.json"),
            "source_run": str(root),
            "runtime": runtime_record(),
            "files": files(output),
        },
    )
    print(json.dumps({"status": status, "output": str(output), "statistics": stats}, ensure_ascii=False, indent=2))
    return 0 if episodes else 2


def doctor():
    print(json.dumps(runtime_record(), ensure_ascii=False, indent=2))
    if sys.version_info < (3, 12):
        raise ValueError("Python 3.12+ required by the fixed Harbor version")
    import harbor

    import scienceinfra

    print("harbor:", harbor.__file__)
    print("scienceinfra:", scienceinfra.__file__)
    for command in [
        ["harbor", "--version"],
        ["docker", "info"],
        ["docker", "compose", "version"],
        ["docker", "buildx", "version"],
    ]:
        result = subprocess.run(command, timeout=40)
        if result.returncode:
            raise ValueError("Dependency check failed: " + shlex.join(command))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect Python imports and CPU Docker; no model call")
    init = commands.add_parser("init", help="Stage a fresh LAPS task plan from ScienceInfra Parquet")
    init.add_argument("--dataset", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--limit", type=int, default=1)
    init.add_argument("--attempts", type=int, default=4)
    init.add_argument("--concurrency", type=int, default=2)
    init.add_argument("--api-base", required=True, help="Your OpenAI-compatible API base URL")
    init.add_argument("--model", default="GLM-5.3", help="Model ID accepted by your API service (example: GLM-5.3)")
    init.add_argument("--max-turns", type=int, default=64)
    init.add_argument("--max-output-tokens", type=int, default=8192)
    init.add_argument("--context-tokens", type=int, default=65536)
    for name in ["preflight", "anchor", "rollout", "review", "export"]:
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name == "review":
            command.add_argument(
                "--retry-replay",
                action="store_true",
                help="Archive failed CPU-only source replay evidence and retry; no teacher calls",
            )
    args = parser.parse_args(argv)
    if args.command == "doctor":
        doctor()
    elif args.command == "init":
        if args.max_turns < 2 or args.max_output_tokens < 1 or args.context_tokens <= args.max_output_tokens:
            parser.error("Invalid turn/output/context budgets")
        init_run(args)
    else:
        root = args.run_dir.resolve()
        with lock(root):
            plan = load_plan(root)
            if args.command == "preflight":
                preflight(root, plan)
            elif args.command == "anchor":
                anchors(root, plan)
            elif args.command == "rollout":
                collect(root, plan)
            elif args.command == "review":
                reviews(root, plan, args.retry_replay)
            elif args.command == "export":
                return export_release(root, plan, args.output.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
