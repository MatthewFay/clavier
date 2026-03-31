import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import BpeTrainer

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PROJECT_ROOT: Path = SCRIPT_DIR.parent
DATA_FILE: Path = PROJECT_ROOT / "data" / "processed" / "bach" / "bach.jsonl"
OUTPUT_FILE: Path = PROJECT_ROOT / "tokenizer.json"

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|bach|>",
    "<|mozart|>",
]


def batch_iterator(batch_size: int = 1000) -> Iterator[list[str]]:
    """Yields batches of clean ABC text from the JSONL file."""
    with open(DATA_FILE, encoding="utf-8") as f:
        batch: list[str] = []
        for line in f:
            data: dict[str, Any] = json.loads(line)
            batch.append(str(data.get("text", "")))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def main() -> None:
    print("Initializing ABC-Aware BPE Tokenizer...")

    tokenizer: Any = Tokenizer(BPE(unk_token="<|unk|>"))

    # --- THE MAGIC: ABC-AWARE PRE-TOKENIZATION ---
    abc_pattern = Regex(
        r"<\|[^|]+\|>|\[[^\]]+\]|![^!]+!|"
        r"[a-zA-Z]|\d|\||\^|_|=|,|'|\/|\(|\)|-"
    )

    tokenizer.pre_tokenizer = Sequence([Split(abc_pattern, behavior="isolated")])

    trainer: Any = BpeTrainer(
        vocab_size=2000,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Training tokenizer on {DATA_FILE}...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Save the compiled tokenizer
    tokenizer.save(str(OUTPUT_FILE))

    print("========================================")
    print(f"Training Complete! Vocab Size: {tokenizer.get_vocab_size()}")
    print(f"Tokenizer saved to: {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()
