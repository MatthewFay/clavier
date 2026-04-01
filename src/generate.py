import argparse
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Clavier

# --- Path Configuration ---
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
CHECKPOINT_PATH = Path("/content/drive/MyDrive/clavier_checkpoints/clavier_best.pt")
OUTPUT_DIR = PROJECT_ROOT / "generated"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_music(
    model: Clavier,
    tokenizer: Any,  # Use Any here to bypass strict checking on the Rust object
    composer: str = "bach",
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generates an ABC music sequence autoregressively."""
    model.eval()

    # 1. Setup the prompt
    prompt = f"<|bos|><|{composer}|>"

    encoded = tokenizer.encode(prompt)
    prompt_ids = cast(list[int], encoded.ids)

    # Context window tensor (Shape: [1, Sequence Length])
    idx: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Generating with temperature {temperature} and top_k {top_k}...")

    block_size = int(model.block_size)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block size if we exceed the model's context window
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            # Forward pass
            logits, _ = model(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature

            # Top-K filtering (prevent the model from picking wildly wrong tokens)
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            idx = torch.cat((idx, next_token), dim=1)

            # Stop if the model naturally predicts the End of Sequence token
            next_token_id = int(next_token.item())

            # Cast the return of decode to a string
            token_str = str(cast(str, tokenizer.decode([next_token_id])))
            if "<|eos|>" in token_str:
                break

    # 2. Decode the final sequence
    generated_ids = cast(list[int], idx[0].tolist())  # type: ignore

    # Cast the return of decode to a string
    raw_text = str(
        cast(str, tokenizer.decode(generated_ids, skip_special_tokens=False))
    )

    # 3. Clean up the output for standard ABC parsers
    clean_text = (
        raw_text.replace("<|bos|>", "")
        .replace(f"<|{composer}|>", "")
        .replace("<|eos|>", "")
        .strip()
    )

    # Add an ABC header if the model didn't generate one (helps with rendering)
    if not clean_text.startswith("X:1"):
        clean_text = (
            f"X:1\nT:Generated {composer.capitalize()} Composition\n{clean_text}"
        )

    return clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate music with Clavier")
    parser.add_argument(
        "--composer", type=str, default="bach", help="Composer token to start with"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.1,
        help="Temperature (higher = more creative/chaotic)",
    )
    parser.add_argument(
        "--length", type=int, default=1000, help="Max tokens to generate"
    )
    parser.add_argument(
        "--out", type=str, default="composition_01.abc", help="Output filename"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = str(args.out)
    out_path = OUTPUT_DIR / out_filename

    # Load Tokenizer using cast to bypass type errors
    tokenizer = cast(Any, Tokenizer.from_file(str(SRC_DIR / "tokenizer.json")))  # type: ignore
    vocab_size = int(cast(int, tokenizer.get_vocab_size()))

    # Load Model & Weights
    print("Loading model weights...")
    model = Clavier(vocab_size=vocab_size, block_size=256).to(device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Could not find weights at {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate
    abc_data = generate_music(
        model,
        tokenizer,
        composer=str(args.composer),
        temperature=float(args.temp),
        max_new_tokens=int(args.length),
    )

    # Save to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(abc_data)

    print(f"\nSuccess! Saved raw ABC to {out_path}")


if __name__ == "__main__":
    main()
