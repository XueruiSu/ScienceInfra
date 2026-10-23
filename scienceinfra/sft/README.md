# Experimental multi-turn SFT collection and export

[简体中文](README.zh-CN.md)

Collect scientific repair demonstrations with a remote teacher, review their
evidence, and export complete multi-turn conversations plus optional Qwen token
labels. This is an initial implementation of trajectory export and SFT data
preparation. It does not run SFT or RL training.

The execution core remains ScienceInfra's task packages, Harbor's **Terminus-2**,
and the task's scientific verifier. This module adds collection scheduling,
strict message/ATIF alignment, native reasoning preservation, source replay,
supervision masks, release manifests, and statistics. It does not use PSRL or
reimplement the agent loop. The model runs at the API provider; Docker runs the
scientific code and verifier on the execution host.

## Scope and compatibility

| Component | Supported contract |
|---|---|
| Task bank | CPU `repair/easy` LAPS tasks from the bundled `demo` |
| Task reference | ScienceInfra commit `49bd221bb18077a349e7f860569c3c5f5a4de64f` |
| Harness | Harbor 0.23.0, reference commit `3c82380859d187957cfd5cd64802b076d9779550` |
| Teacher example | GLM-5.3 through an OpenAI-compatible HTTPS API |
| Teacher evidence | Provider-native `reasoning_content`, raw assistant content and terminal observations |
| Replay adapter | Pinned LAPS `solve.sh`; same source file set, source edits only |
| Target encoding | Qwen thinking template with `enable_thinking` and `preserve_thinking`, immutable tokenizer revision |
| Execution host | Linux amd64, Python 3.12+, Docker Engine, Compose and Buildx |

The reference commits identify the tested contracts, not an assertion that a
modified checkout has those exact contents. Runs hash all SFT Python modules,
task files and evidence. Retain the Git revision/diff and dependency inventory.
The API model alias does not attest immutable provider weights; exports record
`teacher_revision=unattested`.

The replay adapter is **not generic across ScienceIDE environments**. New or
deleted source files, build-system edits and other solver layouts require a new
adapter. Apple Silicon image adaptations from earlier local experiments are not
included. Summarized, branched, continued or misaligned conversations are rejected;
the exporter never guesses missing turns or reconstructs reasoning.

## 1. Install in a fresh checkout

Check out the revision containing this module. The following commands use Bash
and start in that checkout's repository root:

```bash
export SCIENCEINFRA_ROOT="$PWD"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'harbor @ git+https://github.com/harbor-framework/harbor.git@3c82380859d187957cfd5cd64802b076d9779550'
python -m pip install -e '.[sft]'
python -m pip check
python -m scienceinfra.sft.rollout doctor
```

`doctor` checks imports and Docker, without calling a teacher. Install Docker
using its [Linux installation instructions](https://docs.docker.com/engine/install/)
and give the execution user access before proceeding. No local model, GPU, PSRL
or vLLM is required. Image construction needs internet access to download pinned
scientific source and build dependencies. The staged **agent runtime** is set to
`no-network`; the host process still reaches the teacher API.

## 2. Build fresh task packages and the dataset

```bash
bash scripts/prepare/prepare_all.sh \
  --repo "$SCIENCEINFRA_ROOT/demo" --envs laps \
  --difficulty easy --categories repair \
  --data-root "$SCIENCEINFRA_ROOT/data" --stages compile,lines,dataset
export TASK_PARQUET="$SCIENCEINFRA_ROOT/data/laps/repair_easy/train/L2.parquet"
```

These stages compile sparse tasks, resolve source line hints and build Parquet.
The referenced task bank has 32 easy repair candidates, split into 24 train and
8 eval rows. Use the **train** split for collection and reserve eval for evaluation.
The exporter does not create a new leakage-resistant split for arbitrary input.
Parquet stores absolute task paths, so regenerate it on the execution machine.

L2 includes the defect file and description. L1 additionally uses resolved line
information; L3 adds no localization hint. The collector reads the task's
instruction and `extra_info.hint`, rather than replacing the prompt with the
Parquet `prompt` column. Exports retain the hint level. See the
[dataset guide](../datasets/README.md) for preparation details.

## 3. Run a single task

In `python -m scienceinfra.sft.rollout xxx`, `xxx` selects the subcommand and
execution stage. `doctor` does not take `--run-dir`; the other stages share the
same run directory.

| `xxx` | Purpose and main outputs | Calls the teacher API |
|---|---|---|
| `doctor` | Check Python, Harbor and Docker dependencies. Inspect services without launching a task or generating a trajectory. | No |
| `init` | Read task Parquet, select and copy task packages, and save model/attempt/concurrency settings and file hashes in `plan.json`. Require a new run directory; do not launch Docker. | No |
| `anchor` | Run the official oracle repair and nop defective baseline through Harbor/Docker. Require repair rewards 1 and 0 with consistent measured floors; save evidence and defective source. These runs do not consume teacher attempt slots. | No |
| `preflight` | Send a small request to the configured model, require nonempty content and native `reasoning_content`, and save the response and receipt. This checks the API; it is not a task trajectory or an SFT example. | Yes, billed |
| `rollout` | Start Harbor Terminus-2 according to the planned tasks, attempts and concurrency. Let the teacher inspect, edit, compile and test code over multiple turns; save raw messages, ATIF, logs, source artifacts and verifier results, and reconcile existing runs. | Yes, billed |
| `review` | Check scientific scores, runtime exceptions and message/ATIF alignment; assign assistant supervision masks. Run independent CPU Docker replay for candidate source and record admission or exclusion reasons. | No |
| `export` | Recheck reviewed evidence and write accepted complete conversations to `sft/train.jsonl` in a new `--output` directory, together with attempt records, statistics and a hash manifest. Do not launch Docker or generate new answers. | No |

Follow `init → anchor → preflight → rollout → review → export` in the example
below. `--attempts 1` in `init` schedules one later teacher attempt; it does not
generate data immediately. `export` writes message-format SFT data; the separate
encoding command in section 7 produces Qwen token labels.

Choose new output directories; existing runs are immutable configurations.
Set `TEACHER_API_BASE` to your service's base URL and `TEACHER_MODEL` to the model
ID it accepts. `https://api.example.com/v1` is a placeholder and `GLM-5.3` is a
generic example; the exact model ID depends on your deployment. `--api-base` is
required. Pass the service's model ID without an extra `openai/` prefix; the
collector adds that adapter prefix for Harbor.

Keep private service names, endpoints, routing identifiers and credentials out of
public commits. Run plans and receipts retain the actual API configuration, and
exports retain the teacher model ID; review generated artifacts before publishing
them and retain the original evidence privately.

```bash
export TEACHER_API_BASE="https://api.example.com/v1"
export TEACHER_MODEL="GLM-5.3"
export RUN_DIR="$SCIENCEINFRA_ROOT/outputs/sft/smoke"
export RELEASE_DIR="$SCIENCEINFRA_ROOT/outputs/sft/smoke-release"
python -m scienceinfra.sft.rollout init \
  --dataset "$TASK_PARQUET" --run-dir "$RUN_DIR" \
  --limit 1 --attempts 1 --concurrency 1 \
  --api-base "$TEACHER_API_BASE" \
  --model "$TEACHER_MODEL"
python -m scienceinfra.sft.rollout anchor --run-dir "$RUN_DIR"
```

`anchor` runs the official fix (`oracle`) and the defective baseline (`nop`).
Oracle must pass all task checks with repair reward 1; nop must have repair
reward 0 and a consistent measured floor. It also captures the defective source
for replay comparisons. Neither step calls the teacher.

Read the key into the environment without writing it into commands or plans:

```bash
read -r -s -p 'Model API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
python -m scienceinfra.sft.rollout preflight --run-dir "$RUN_DIR"
python -m scienceinfra.sft.rollout rollout --run-dir "$RUN_DIR"
unset OPENAI_API_KEY
python -m scienceinfra.sft.rollout review --run-dir "$RUN_DIR"
python -m scienceinfra.sft.rollout export \
  --run-dir "$RUN_DIR" --output "$RELEASE_DIR"
```

`unset OPENAI_API_KEY` is a shell command, not a `rollout` subcommand. It removes
the variable from the current shell so subsequently launched processes do not
inherit it; it does not revoke the provider key or delete run records. After
collection, `review` and `export` do not need a teacher API key.

**Preflight and rollout incur provider charges.** Preflight requires a nonempty
native `reasoning_content` field and saves the response. A successful preflight
does not guarantee a successful task. Review runs CPU Docker source replay,
without teacher calls. Export is offline.

The conversation preserves Terminus-2's JSON text protocol: terminal feedback is
a `user` message; actions remain `assistant.content`. They are not rewritten as
native function calls. Provider reasoning remains `reasoning_content`.

## 4. Admission and source replay

An episode is exported only when all applicable checks pass:

1. A finished Harbor trial has no exception; its result and raw verifier reward
   agree, with a valid floor and repair normalization.
2. Reward, repair reward, equivalence and all task-specific checks pass; the
   grading report confirms each required check.
3. Messages match raw ATIF assistant text, initial input and intervening terminal
   feedback exactly. There is one uncompressed trajectory, ending with the two
   completion confirmations expected by Terminus-2.
4. At least two assistant turns are eligible for supervision, including an action
   turn. Missing native reasoning, protocol warnings and explicit terminal errors
   make a turn context-only (`loss=false`) while preserving its original text.
5. Captured scientific source differs from the defective baseline. Replay imports
   **model-produced source** into a fresh defective task, removes the reference
   fix, clears compiled artifacts, and uses the known build/check recipe. The
   fresh verifier must pass again.

This is automated evidence review, not scientific expert review. Command-pattern
checks and error masking are conservative heuristics. A replay verifies the
captured source with the reference build recipe; it does not prove every recorded
action was necessary or that arbitrary model-written build scripts are correct.
Checksums detect changes relative to retained receipts, not malicious rewriting
of the entire evidence chain.

## 5. Repeat attempts and resume safely

After the smoke run, create a separate batch:

```bash
export RUN_DIR="$SCIENCEINFRA_ROOT/outputs/sft/batch-4x"
export RELEASE_DIR="$SCIENCEINFRA_ROOT/outputs/sft/batch-4x-release"
python -m scienceinfra.sft.rollout init \
  --dataset "$TASK_PARQUET" --run-dir "$RUN_DIR" \
  --limit 4 --attempts 4 --concurrency 2 \
  --api-base "$TEACHER_API_BASE" --model "$TEACHER_MODEL" \
  --max-turns 64 --max-output-tokens 8192 --context-tokens 65536
```

Repeat `anchor`, key entry, `preflight`, `rollout`, `review`, and `export` from the
previous section with these directories. This schedules 16 independent attempts,
at most two at once. Oracle/nop and review are sequential. Docker compilations
also consume CPU and memory, so scale concurrency according to the host.
`context-tokens` describes model metadata; it is not a guaranteed input-token cap.

- Rerun `rollout` on the same run to reconcile saved results. Live original
  children block duplicate execution. Interrupted attempts consume their slot
  and are preserved; they are not silently repeated as new paid attempts.
- Harbor trial retries and provider transport retries are disabled. Protocol
  recovery within the harness can still create additional calls in one episode.
- Provider authentication/model/quota errors stop unstarted work and write
  `api_blocker.json`. Diagnose the problem and preserve that file under a history
  directory before retrying; already running attempts may still finish.
- To explicitly retry a failed **CPU source replay**, use
  `python -m scienceinfra.sft.rollout review --run-dir "$RUN_DIR" --retry-replay`.
  The failed replay and review are archived; this makes no new teacher call.
- Changing settings, tasks or SFT source code requires a new run. Export to a new
  directory each time. An empty admitted dataset returns exit code 2, with an
  exclusion report rather than fabricated examples.

## 6. Artifacts and statistics

```text
<run>/
  plan.json, plan.sha256      # settings, source hashes, staged task provenance
  tasks/, anchors/           # runnable task copies and oracle/nop evidence
  preflight/                 # provider-native reasoning probe
  attempts/<task>/<slot>/    # raw Harbor results, ATIF, terminal logs, source
                             # artifacts, review and independent source replay
<release>/
  sft/train.jsonl            # one complete conversation per admitted episode
  attempts.jsonl            # every planned slot, including failures
  tasks.jsonl, evidence.json
  statistics.json, manifest.json
```

`statistics.json` reports scientific success and SFT admission separately, with
ended attempts as denominator, distinct admitted tasks, exclusions and conversation
round distributions. User rounds count the initial task and automated terminal
feedback, not human messages. API input/output totals sum requests and include
repeated context; they are **not** target-tokenized sequence lengths and exclude
preflight usage. Keep raw runs with releases for audit. Generated runs may contain
local paths, source code and provider output; they are not included in this PR.

## 7. Optional Qwen labels

Use an immutable 40-character Hugging Face tokenizer commit selected for the
target model. Only tokenizer assets are downloaded, not model weights:

```bash
python -m pip install -e '.[sft-qwen]'
export QWEN_MODEL='Qwen/Qwen3.8-27B'
export QWEN_REV='<replace-with-verified-40-character-tokenizer-commit>'
python -m scienceinfra.sft.tokenize_qwen \
  --release-dir "$RELEASE_DIR" \
  --output-dir "$SCIENCEINFRA_ROOT/outputs/sft/batch-4x-qwen" \
  --model "$QWEN_MODEL" --revision "$QWEN_REV" --max-tokens 65536
```

With all assets for that revision cached, add `--local-files-only`. Encoding checks
the release hashes and template/token prefix boundaries, preserves reasoning in
all assistant turns, and supervises only reviewed assistant spans. User context
and excluded assistant turns have `labels=-100`. Overlength whole episodes are
excluded without truncation. An incompatible template fails rather than silently
dropping reasoning. This is a template contract, not universal support for every
Qwen release.

Output contains `train.jsonl` (`input_ids`, `attention_mask`, `labels`, supervised
spans), saved tokenizer assets and `report.json` with sequence/supervised token
statistics, exclusions, revisions and hashes. A downstream trainer must preserve
these labels, use a compatible collator and perform its own integration checks.
This PR prepares training inputs; it does not implement a trainer integration.
Teacher demonstrations are not on-policy token/logprob records for PSRL GRPO.

## Development and validation

```bash
python -m pip install ruff==0.15.6
python -m unittest discover -s tests/sft -p 'test_*.py' -v
ruff check scienceinfra/sft tests/sft
ruff format --check scienceinfra/sft tests/sft
```

The tests use synthetic conversation/evidence fixtures and mock replay execution;
they make no Docker or model calls. An optional cached-tokenizer test can be
enabled with `SFT_QWEN_TOKENIZER_DIR`. See [VALIDATION.md](VALIDATION.md) for the
actual validation performed and the end-to-end checks still required.
