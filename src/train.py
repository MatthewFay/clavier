from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from dataset import ABCMusicDataset

# Import the architecture and dataset we built
from model import Clavier

# --- Configuration ---
BATCH_SIZE = 64
BLOCK_SIZE = 256
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
SAVE_INTERVAL = 5  # Save a checkpoint every 5 epochs

# Assuming you mounted Google Drive in Colab:
# from google.colab import drive
# drive.mount('/content/drive')
CHECKPOINT_DIR = Path("/content/drive/MyDrive/clavier_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "clavier_latest.pt"

# --- Setup Device ---
# Colab T4 GPU, or fall back to Mac CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
) -> None:
    """Saves the complete state required to resume training."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Pylance Fix: Explicitly type the dictionary
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\n---> Checkpoint saved at Epoch {epoch} to {CHECKPOINT_PATH}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Loads state if a checkpoint exists, returns the epoch to resume from."""
    if CHECKPOINT_PATH.exists():
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")

        # Pylance Fix: Cast the loaded object to a typed dictionary
        checkpoint: dict[str, Any] = cast(
            dict[str, Any],
            torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch: int = int(checkpoint["epoch"]) + 1
        prev_loss: float = float(checkpoint["loss"])

        # Ruff Fix: Break string across lines
        print(f"Resuming from Epoch {start_epoch} (Previous Loss: {prev_loss:.4f})")
        return start_epoch

    print("No checkpoint found. Starting training from scratch.")
    return 0


def main() -> None:
    # 1. Load Data
    dataset = ABCMusicDataset(
        data_path=Path("data/processed/bach/bach.jsonl"),
        tokenizer_path=Path("src/tokenizer.json"),
        block_size=BLOCK_SIZE,
    )

    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # 2. Initialize Model and Optimizer
    vocab_size = dataset.tokenizer.get_vocab_size()
    model = Clavier(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Handle Checkpointing
    start_epoch = load_checkpoint(model, optimizer)

    # 4. The Training Loop
    model.train()
    for epoch in range(start_epoch, MAX_EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader):
            # Move tensors to the GPU
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)

            # Pylance Fix: Use `_` for logits since we only need the loss for training
            _, loss = model(x, targets=y)

            # PyTorch safely expects loss to not be None when targets are provided
            assert loss is not None

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Pylance Fix: Cast optimizer to Any to clear untyped .step() method
            cast(Any, optimizer).step()
            total_loss += loss.item()

            # Print step progress every 50 batches
            # Ruff Fix: Break string across lines
            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        # Epoch Complete
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ===")

        # Save checkpoint periodically
        if (epoch % SAVE_INTERVAL == 0) or (epoch == MAX_EPOCHS - 1):
            save_checkpoint(model, optimizer, epoch, avg_loss)


if __name__ == "__main__":
    main()
