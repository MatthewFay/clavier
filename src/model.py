import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    The core mechanism where notes 'look back' at previous notes
    to figure out harmony and timing.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch Size, Sequence Length (Block Size), Embedding Size

        # Calculate Query, Key, and Value for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention: (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # PyTorch 2.0+ Flash Attention (Incredibly fast, mathematically equivalent)
        # is_causal=True ensures notes cannot "look ahead" into the future.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.resid_dropout(self.c_proj(y))


class TransformerBlock(nn.Module):
    """A single layer of the Transformer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)

        # A simple FeedForward network to let the model "think"
        # about what it attended to
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Clavier(nn.Module):
    """The full Language Model Architecture."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # 1. Embeddings: Convert integers into rich numerical vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # 2. The Transformer Blocks (The deep "thought" layers)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # 3. Final normalization and output classification head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: Share weights between token embedding and output layer
        # (saves parameters)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT weight initialization strategy."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()  # type: ignore
        assert T <= self.block_size, f"""Cannot forward sequence of length {T},
            block size is only {self.block_size}"""

        # Calculate positional and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos_emb = self.position_embedding(pos)  # (T, d_model)

        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # PyTorch's cross_entropy expects (Batch * Sequence, Classes)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Generates new music tokens autoregressively."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            # Get predictions for the next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx


def test_model() -> None:
    """Proves the model can accept our dataloader tensors and output a loss."""
    vocab_size = 950  # From tokenizer
    block_size = 256
    batch_size = 4

    print("Initializing Clavier...")
    model = Clavier(vocab_size=vocab_size, block_size=block_size)

    # Simulate a batch of data from DataLoader
    xb = torch.randint(0, vocab_size, (batch_size, block_size))
    yb = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(xb, yb)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"Output Logits Shape: {logits.shape} -> (Batch, Time, Vocab Size)")
    print(f"Initial Random Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_model()
