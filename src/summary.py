from pathlib import Path
from typing import Any, cast

import torch
from tokenizers import Tokenizer
from torchinfo import summary

from model import Clavier

# --- Path Configuration ---
SRC_DIR = Path(__file__).parent.resolve()
TOKENIZER_PATH = SRC_DIR / "tokenizer.json"


def main() -> None:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Could not find tokenizer at {TOKENIZER_PATH}")

    # 1. Get exact vocab size from your trained tokenizer
    tokenizer = cast(Any, Tokenizer.from_file(str(TOKENIZER_PATH)))  # type: ignore
    vocab_size = int(cast(int, tokenizer.get_vocab_size()))
    block_size = 256

    print(f"Loaded Tokenizer with Vocab Size: {vocab_size}")
    print("Initializing Clavier architecture...\n")

    # 2. Initialize a blank model (no weights needed for a summary)
    model = Clavier(vocab_size=vocab_size, block_size=block_size)

    # 3. Generate the summary
    # We simulate a single sequence passing through the model: Batch Size 1, Length 256
    # dtypes=[torch.long] is critical because the model expects integer token IDs,
    # not floats
    model_summary = summary(
        model,
        input_size=(1, block_size),
        dtypes=[torch.long],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0,  # Suppress default print so we can print cleanly
    )

    print(model_summary)


if __name__ == "__main__":
    main()
