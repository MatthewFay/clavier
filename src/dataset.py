import json
from pathlib import Path
from typing import Any, cast

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PROJECT_ROOT: Path = SCRIPT_DIR.parent

TOKENIZER_FILE: Path = SCRIPT_DIR / "tokenizer.json"
DATA_FILE: Path = PROJECT_ROOT / "data" / "processed" / "bach" / "bach.jsonl"


class ABCMusicDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_path: Path, tokenizer_path: Path, block_size: int = 512):
        """
        Initializes the dataset by loading all JSONL records, tokenizing them,
        and concatenating them into a single continuous 1D tensor.
        """
        self.block_size = block_size

        print("Loading Tokenizer...")
        self.tokenizer: Any = Tokenizer.from_file(str(tokenizer_path)) # type: ignore

        print(f"Loading and Tokenizing data from {data_path.name}...")
        all_tokens: list[int] = []

        with open(data_path, encoding="utf-8") as f:
            for line in f:  # jsonl files are line-delimited JSON,
                            # so we read line by line
                record: dict[str, Any] = json.loads(line)
                text: str = str(record.get("text", ""))

                # Encode the string to a list of integer IDs
                encoded: Any = self.tokenizer.encode(text) # type: ignore

                ids: list[int] = cast(list[int], encoded.ids) # type: ignore
                all_tokens.extend(ids)

        # Convert the massive Python list into a highly optimized PyTorch LongTensor
        self.data: torch.Tensor = torch.tensor(all_tokens, dtype=torch.long)

        print(f"Dataset compiled! Total tokens: {len(self.data):,}")
        print(f"Total trainable windows (block_size={block_size}): {len(self):,}")

    def __len__(self) -> int:
        # We subtract block_size so we don't grab a window that goes out of bounds
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Grabs a slice of tokens for the input (X) and the target (Y).
        In autoregressive training, Y is just X shifted one token to the right.
        """
        # Input sequence
        x: torch.Tensor = self.data[idx : idx + self.block_size]

        # Target sequence (what the model needs to predict)
        y: torch.Tensor = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y


def test_dataloader() -> None:
    """Run this file directly to test the DataLoader output."""
    block_size: int = 256
    batch_size: int = 4

    dataset = ABCMusicDataset(DATA_FILE, TOKENIZER_FILE, block_size=block_size)

    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Grab the very first batch
    xb, yb = next(iter(dataloader))

    print("\n" + "=" * 50)
    print(f"Input Tensor Shape (X): {xb.shape} -> (Batch Size, Block Size)")
    print(f"Target Tensor Shape (Y): {yb.shape} -> (Batch Size, Block Size)")
    print("=" * 50)

    # Decode the very first sequence in the batch back to text to prove it works
    sample_ids: list[int] = xb[0].tolist()

    decoded_text: str = str(dataset.tokenizer.decode(sample_ids))

    print("\nSample Decoder Test (First sequence in batch):")
    print(decoded_text[:200] + " ... [Truncated for display]")


if __name__ == "__main__":
    test_dataloader()
