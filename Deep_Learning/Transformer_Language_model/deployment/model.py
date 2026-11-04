
import json
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
TOKENIZER_DIR = ARTIFACT_DIR / "tokenizer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Attention
# -------------------------

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_length = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )

        return self.dropout(attention_output)


# -------------------------
# Feed Forward
# -------------------------

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


# -------------------------
# Transformer Block
# -------------------------

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


# -------------------------
# Transformer Language Model
# -------------------------

class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        block_size,
        dropout
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):
        _, seq_length = x.shape

        token_embeddings = self.token_embedding(x)

        positions = torch.arange(
            seq_length,
            device=x.device
        )

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        return self.output_layer(x)


# -------------------------
# Load model and tokenizer
# -------------------------

with open(ARTIFACT_DIR / "config.json", "r") as f:
    CONFIG = json.load(f)

tokenizer = ByteLevelBPETokenizer(
    str(TOKENIZER_DIR / "vocab.json"),
    str(TOKENIZER_DIR / "merges.txt")
)

model = TransformerLanguageModel(
    vocab_size=CONFIG["vocab_size"],
    embedding_dim=CONFIG["embedding_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"],
    block_size=CONFIG["block_size"],
    dropout=CONFIG["dropout"]
).to(DEVICE)

checkpoint = torch.load(
    ARTIFACT_DIR / "transformer_best.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()


# Special tokens
PAD_ID = CONFIG["pad_id"]
BOS_ID = CONFIG["bos_id"]
EOS_ID = CONFIG["eos_id"]
BOY_ID = CONFIG["boy_id"]
GIRL_ID = CONFIG["girl_id"]


# -------------------------
# Girl response generation
# -------------------------

def generate_girl_response(
    conversation_history,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
):

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    model.eval()

    input_ids = []

    for speaker, message in conversation_history:

        if speaker == "BOY":
            input_ids.append(BOY_ID)

        elif speaker == "GIRL":
            input_ids.append(GIRL_ID)

        else:
            raise ValueError(f"Unknown speaker: {speaker}")

        input_ids.extend(tokenizer.encode(message).ids)
        input_ids.append(EOS_ID)

    # Request a girl response
    input_ids.append(GIRL_ID)

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = []

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = input_ids[:, -CONFIG["block_size"]:]

            logits = model(context)
            logits = logits[:, -1, :]

            logits = logits / temperature

            # Block special tokens that should not appear in replies
            for token_id in [BOY_ID, GIRL_ID, PAD_ID, BOS_ID]:
                logits[:, token_id] = float("-inf")

            current_top_k = min(top_k, logits.size(-1))

            values, indices = torch.topk(
                logits,
                current_top_k
            )

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )

            filtered_logits.scatter_(1, indices, values)

            probabilities = torch.softmax(
                filtered_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token_id = next_token.item()

            if token_id == EOS_ID:
                break

            generated_ids.append(token_id)

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

    return tokenizer.decode(generated_ids)
