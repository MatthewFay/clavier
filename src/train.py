from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from dataset import ABCMusicDataset
from model import Clavier

# --- Path Configuration ---
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent

# --- Configuration ---
BATCH_SIZE = 64
BLOCK_SIZE = 256
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100

# Step-Based Evaluation Configuration
EVAL_INTERVAL = 3000  # Run validation and save every 3000 steps
EVAL_ITERS = 200  # How many batches to sample for a fast validation estimate
PATIENCE = 5  # Stop if validation doesn't improve for 5 intervals (15k steps)

# Assuming mounted Google Drive in Colab:
# from google.colab import drive
# drive.mount('/content/drive')
CHECKPOINT_DIR = Path("/content/drive/MyDrive/clavier_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "clavier_latest.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "clavier_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    is_best: bool = False,
) -> None:
    """Saves the complete state required to resume training."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "step": step,  # Added step tracking for precise resuming
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # Always save the latest for resuming
    torch.save(checkpoint, CHECKPOINT_PATH)

    # If it's a new record, save a protected copy
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"\n---> New Best Model saved at Epoch {epoch}, Step {step}")


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[int, int, float]:
    """Loads state if a checkpoint exists, returns (epoch, step, best_loss)."""
    if CHECKPOINT_PATH.exists():
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")

        checkpoint: dict[str, Any] = cast(
            dict[str, Any],
            torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch: int = int(checkpoint["epoch"])
        start_step: int = int(
            checkpoint.get("step", 0)
        )  # Default to 0 if old checkpoint
        prev_loss: float = float(checkpoint["loss"])

        print(
            f"""Resuming from Epoch {start_epoch},
            Step {start_step} (Prev Loss: {prev_loss:.4f})"""
        )
        return start_epoch, start_step, prev_loss

    print("No checkpoint found. Starting training from scratch.")
    return 0, 0, float("inf")


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, dataloader: DataLoader[Any], eval_iters: int
) -> float:
    """Evaluates the model on a random subset of unseen data for speed."""
    model.eval()
    total_loss = 0.0

    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        assert loss is not None
        total_loss += loss.item()

    model.train()
    # Prevent division by zero if eval_iters is larger than the dataset
    actual_iters = min(eval_iters, len(dataloader))
    return total_loss / actual_iters


def main() -> None:
    # 1. Load Training Data
    train_dataset = ABCMusicDataset(
        data_path=PROJECT_ROOT / "data/processed/bach/bach_train.jsonl",
        tokenizer_path=SRC_DIR / "tokenizer.json",
        block_size=BLOCK_SIZE,
    )
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # 2. Load Validation Data
    val_dataset = ABCMusicDataset(
        data_path=PROJECT_ROOT / "data/processed/bach/bach_val.jsonl",
        tokenizer_path=SRC_DIR / "tokenizer.json",
        block_size=BLOCK_SIZE,
    )
    # Changed shuffle=True so estimate_loss gets a different random sample every time
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    vocab_size = train_dataset.tokenizer.get_vocab_size()
    model = Clavier(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_epoch, start_step, best_val_loss = load_checkpoint(model, optimizer)
    intervals_without_improvement = 0

    # The Training Loop
    model.train()
    for epoch in range(start_epoch, MAX_EPOCHS):
        for step, (x, y) in enumerate(train_dataloader):
            # Fast-forward dataloader if resuming mid-epoch
            if epoch == start_epoch and step < start_step:
                continue

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

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(train_dataloader)} "
                    f"| Train Loss: {loss.item():.4f}"
                )

            # ---> Step-Based Evaluation & Checkpointing <---
            if step > 0 and step % EVAL_INTERVAL == 0:
                print(f"\n--- Running Evaluation at Step {step} ---")
                val_loss = estimate_loss(model, val_dataloader, EVAL_ITERS)
                print(f"Validation Loss Estimate: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    intervals_without_improvement = 0
                    save_checkpoint(
                        model, optimizer, epoch, step, val_loss, is_best=True
                    )
                else:
                    intervals_without_improvement += 1
                    print(
                        f"Warning: Validation loss increased. Patience: "
                        f"{intervals_without_improvement}/{PATIENCE}"
                    )
                    save_checkpoint(
                        model, optimizer, epoch, step, val_loss, is_best=False
                    )

                    if intervals_without_improvement >= PATIENCE:
                        print(
                            f"\nEARLY STOPPING TRIGGERED at Epoch {epoch}, Step {step}."
                        )
                        print(
                            f"""Model overfitted.
                            Revert to {BEST_MODEL_PATH} for best weights."""
                        )
                        return  # Exits the training loop entirely

        # Reset start_step after the first resumed epoch finishes
        start_step = 0
        print(f"=== Epoch {epoch} Completed ===")


if __name__ == "__main__":
    main()
