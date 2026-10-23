# Validation scope

This initial SFT module was assembled and checked on 2026-10-22. It is migrated
from independent scientific-repair data preparation tooling, not a claim to
reproduce the paper's full SFT recipe. Only code, documentation and synthetic
test fixtures are included; historical trajectories and credentials are not.

## Checks performed for this change

| Check | Result |
|---|---|
| Offline contract suite | 57 passed; the optional cached-tokenizer test is skipped unless configured |
| Cached real Qwen tokenizer test | 1 passed separately; no downloads or model weights |
| Ruff lint and format | Passed for `scienceinfra/sft` and `tests/sft` with Ruff 0.15.6 |
| CLI | Both module `--help` entry points passed without Docker or API access |
| Documentation | All 14 Bash blocks in the two module guides passed `bash -n` |
| Packaging | Editable core package installation and `pip check` passed in an isolated environment |
| Clean source compilation | 192 LAPS Harbor task directories compiled from this checkout's tracked task sources |
| Dataset construction | 32 easy repair rows, 24 train and 8 eval, generated afresh for L1/L2/L3 |
| Fresh run initialization | One train L2 task staged; plan and task integrity rechecked successfully |

The offline suite ran with Python 3.13 and the pinned Harbor 0.23.0 source.
Core package installation, task compilation, dataset construction and initialization
used Python 3.11; actually running the pinned Harbor requires Python 3.12+.
Line-hint network resolution was not rerun; dataset construction used the tracked
`DEFECT_LINES.json`. No earlier compiled task directory, dataset or review receipt
was imported into the staging check.

The real-tokenizer check used Transformers 5.8.1 and locally cached assets for
`Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
after verifying their recorded SHA-256 hashes. The dialogue was **synthetic**.
The check verifies that native reasoning in supervised and context-only turns
survives the actual template, and that only reviewed spans receive non-`-100`
labels. It is not a training run or evidence of model-quality improvement.

To repeat the optional tokenizer check after installing `.[sft-qwen]`:

```bash
export SFT_QWEN_TOKENIZER_DIR=/path/to/cached/tokenizer
HF_HUB_OFFLINE=1 python -m unittest discover \
  -s tests/sft -p test_tokenize_real.py -v
```

## Not executed for this PR

- Fresh Linux host setup and installation of the full optional Harbor stack.
- Docker image construction and oracle/nop scientific checks with the packaged module.
- A paid model API preflight or a newly generated teacher rollout.
- Real Docker source replay and admission of a new model-generated episode.
- End-to-end encoding/export of a newly collected episode with this packaged module.
- Model training or downstream trainer/collator integration.

GitHub Actions runs the offline suite, lint and CLI checks on Linux/Python 3.12.
It does not launch Docker tasks, require teacher credentials or call an API.
The remote run's status is reported by the PR checks; the local results above
must not be interpreted as an already completed remote CI run.

Before treating a deployment as operational, follow the one-task workflow in the
module guide and preserve its anchor, preflight, rollout, review and export evidence.
Historical local successes do not replace this check of the newly packaged module.
