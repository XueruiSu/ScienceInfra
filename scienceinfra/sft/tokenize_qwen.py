#!/usr/bin/env python3
"""Optional Qwen thinking-template encoding with explicit assistant loss masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_episode(row, tokenizer, max_tokens):
    messages = row["messages"]
    targets = [i for i, message in enumerate(messages) if message.get("loss") is True]
    if len(messages) < 4 or len(targets) < 2:
        raise ValueError("Episode must have multiple reviewed assistant turns")
    clean = [{k: v for k, v in message.items() if k != "loss"} for message in messages]
    kwargs = {"enable_thinking": True, "preserve_thinking": True}
    full = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False, **kwargs)
    ids = tokenizer(full, add_special_tokens=False).input_ids
    if not 0 < len(ids) <= max_tokens:
        raise ValueError("Complete episode exceeds max_tokens; it was not truncated")
    labels, spans = [-100] * len(ids), []
    for index, message in enumerate(clean):
        if message["role"] != "assistant":
            if index in targets:
                raise ValueError("Only assistant messages may be supervised")
            continue
        reasoning = message.get("reasoning_content", "").strip()
        if index in targets and not reasoning:
            raise ValueError("Supervised target lacks native reasoning")
        prefix = tokenizer.apply_chat_template(clean[:index], tokenize=False, add_generation_prompt=True, **kwargs)
        through = tokenizer.apply_chat_template(
            clean[: index + 1], tokenize=False, add_generation_prompt=False, **kwargs
        )
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        through_ids = tokenizer(through, add_special_tokens=False).input_ids
        start, end = len(prefix_ids), len(through_ids)
        if (
            not full.startswith(through)
            or not through.startswith(prefix)
            or ids[:start] != prefix_ids
            or ids[:end] != through_ids
            or not start < end
            or (reasoning and reasoning not in through[len(prefix) :])
        ):
            raise ValueError("Template drops native reasoning or crosses a loss boundary")
        if index in targets:
            labels[start:end] = ids[start:end]
            spans.append({"message_index": index, "start": start, "end": end})
    return {
        "episode_id": row["episode_id"],
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "supervised_assistant_spans": spans,
    }


def verified_release(root):
    """Require the consumed JSONL to be explicitly covered by the manifest."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "scienceinfra_sft_v1":
        raise ValueError("Unsupported SFT release schema")
    if manifest.get("status") not in {"complete", "partial"}:
        raise ValueError("Release contains no admitted SFT data")
    evidence = manifest.get("files")
    if not isinstance(evidence, dict) or "sft/train.jsonl" not in evidence:
        raise ValueError("Release manifest does not authenticate sft/train.jsonl")
    for rel, expected in evidence.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("Release evidence changed: " + rel)
    rows = [json.loads(line) for line in (root / "sft/train.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No accepted SFT episodes to encode")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", required=True, help="Immutable 40-character HF commit")
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use an immutable Hugging Face commit, not main")
    if args.max_tokens <= 0 or args.output_dir.exists():
        parser.error("Positive token limit and NEW output directory required")
    root = args.release_dir.resolve()
    rows = verified_release(root)
    from transformers import AutoTokenizer, __version__

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template")
    records, excluded = [], []
    for row in rows:
        try:
            records.append(encode_episode(row, tokenizer, args.max_tokens))
        except ValueError as error:
            excluded.append({"episode_id": row["episode_id"], "reason": str(error)})
    args.output_dir.mkdir(parents=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    output = args.output_dir / "train.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    lengths = [len(row["input_ids"]) for row in records]
    supervision = [sum(t != -100 for t in row["labels"]) for row in records]

    def summary(values):
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
        }

    report = {
        "status": "encoded" if records else "no_admissible_sequences",
        "model": args.model,
        "revision": args.revision,
        "transformers_version": __version__,
        "source_release_manifest_sha256": digest(root / "manifest.json"),
        "source_sft_sha256": digest(root / "sft/train.jsonl"),
        "max_tokens": args.max_tokens,
        "sequence_tokens": summary(lengths),
        "supervised_tokens": summary(supervision),
        "excluded": excluded,
        "model_training_performed": False,
        "files": {
            str(p.relative_to(args.output_dir)): digest(p) for p in sorted(args.output_dir.rglob("*")) if p.is_file()
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "sequences": len(records), "excluded": len(excluded)}))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
