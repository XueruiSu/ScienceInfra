"""Optional offline contract check against locally cached Qwen tokenizer assets."""

import os
import unittest

from scienceinfra.sft.tokenize_qwen import encode_episode


@unittest.skipUnless(
    os.environ.get("SFT_QWEN_TOKENIZER_DIR"), "set SFT_QWEN_TOKENIZER_DIR for the real tokenizer check"
)
class CachedQwenTests(unittest.TestCase):
    def test_all_native_thinking_survives_and_only_reviewed_turns_are_supervised(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["SFT_QWEN_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
        )
        row = {
            "episode_id": "synthetic-cached-tokenizer-contract",
            "messages": [
                {"role": "user", "content": "Inspect this synthetic solver fixture.", "loss": False},
                {
                    "role": "assistant",
                    "content": '{"commands": []}',
                    "reasoning_content": "First native reasoning.",
                    "loss": True,
                },
                {"role": "user", "content": "Synthetic terminal feedback.", "loss": False},
                {
                    "role": "assistant",
                    "content": "A context-only response.",
                    "reasoning_content": "Context-only native reasoning.",
                    "loss": False,
                },
                {"role": "user", "content": "Synthetic completion check.", "loss": False},
                {
                    "role": "assistant",
                    "content": '{"task_complete": true}',
                    "reasoning_content": "Final native reasoning.",
                    "loss": True,
                },
            ],
        }
        encoded = encode_episode(row, tokenizer, 4096)
        self.assertEqual([span["message_index"] for span in encoded["supervised_assistant_spans"]], [1, 5])
        decoded = tokenizer.decode(encoded["input_ids"])
        for message in row["messages"]:
            if "reasoning_content" in message:
                self.assertIn(message["reasoning_content"], decoded)
        positions = {
            index for span in encoded["supervised_assistant_spans"] for index in range(span["start"], span["end"])
        }
        self.assertGreater(len(positions), 0)
        for index, label in enumerate(encoded["labels"]):
            self.assertEqual(label, encoded["input_ids"][index] if index in positions else -100)


if __name__ == "__main__":
    unittest.main()
