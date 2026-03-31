from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from dataset import ABCMusicDataset
from model import Clavier

# --- Configuration ---
BATCH_SIZE = 64
BLOCK_SIZE = 256
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
SAVE_INTERVAL = 5  # Save a checkpoint every 5 epochs

# Assuming mounted Google Drive in Colab:
# from google.colab import drive
# drive.mount('/content/drive')
CHECKPOINT_DIR = Path("/content/drive/MyDrive/clavier_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "clavier_latest.pt"

# --- Setup Device ---
# Colab T4 GPU, or fall back to CPU
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

        checkpoint: dict[str, Any] = cast(
            dict[str, Any],
            torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch: int = int(checkpoint["epoch"]) + 1
        prev_loss: float = float(checkpoint["loss"])

        print(f"Resuming from Epoch {start_epoch} (Previous Loss: {prev_loss:.4f})")
        return start_epoch

    print("No checkpoint found. Starting training from scratch.")
    return 0


def main() -> None:
    dataset = ABCMusicDataset(
        data_path=Path("data/processed/bach/bach.jsonl"),
        tokenizer_path=Path("src/tokenizer.json"),
        block_size=BLOCK_SIZE,
    )

    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    vocab_size = dataset.tokenizer.get_vocab_size()
    model = Clavier(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_epoch = load_checkpoint(model, optimizer)

    # The Training Loop
    model.train()
    for epoch in range(start_epoch, MAX_EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader):
            # Move tensors to the GPU
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)

            _, loss = model(x, targets=y)

            assert loss is not None

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            cast(Any, optimizer).step()
            total_loss += loss.item()

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
