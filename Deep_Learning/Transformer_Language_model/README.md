# TwinTalk — a small GPT-style chat model, built from scratch

TwinTalk is a conversational language model I built from scratch in PyTorch. Give it a chat conversation and it writes the next reply.

Nothing here is a pre-trained model. I trained the tokenizer, prepared the data, wrote the transformer, and wrote the training loop myself. That was the whole point — I wanted to understand how models like ChatGPT work from the inside, not just call an API.

## How it works

**1. Data**

I used a dataset of about 123,000 short chat messages between two people. First I cleaned it — removed empty messages, "deleted message" texts, media placeholders and duplicates. About 118,000 messages were left.

I split them in time order (90% train / 5% validation / 5% test), so the model is always tested on newer messages than the ones it trained on.

Each message is marked with a speaker token, `<|boy|>` or `<|girl|>`, so the model learns both sides of the conversation.

**2. Tokenizer**

I trained a byte-level BPE tokenizer on this data itself (vocabulary: 4,000). Special tokens: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|boy|>`, `<|girl|>`.

**3. Model**

A decoder-only transformer — the same basic design as GPT, just very small.

| Setting | Value |
| --- | --- |
| Parameters | 1,432,992 (~1.4M) |
| Embedding size | 128 |
| Layers | 2 |
| Attention heads | 4 |
| Context length | 64 tokens |
| Dropout | 0.2 |
| Vocabulary | 4,000 |

Each block is pre-norm: LayerNorm → causal multi-head self-attention → residual connection, then LayerNorm → feed-forward (128 → 512 → 128, GELU) → residual connection. Token and position embeddings go in at the start, and a linear layer at the end predicts the next token.

**4. Training**

- Loss: cross-entropy · Optimizer: AdamW (learning rate 0.0003, weight decay 0.0001)
- Batch size 32, gradient clipping at 1.0, max 10 epochs, early stopping with patience 2
- The best checkpoint is picked by validation loss, not the last epoch
- Trained on my laptop GPU — NVIDIA RTX 3050 (6 GB)

**5. Generation**

Replies are generated one token at a time, with temperature 0.8 and top-k 40 sampling. Speaker tokens are blocked inside a reply, and generation stops at `<|eos|>`.

## Results

On the test set — messages the model never saw during training:

| Metric | Value |
| --- | --- |
| Test loss | 3.85 |
| Next-token accuracy | 29.6% |
| Perplexity | 46.84 |

This was my third attempt, and the first two taught me more than the third:

1. **Bigram baseline** — simple counts of which token follows which. Something basic to compare against.
2. **LSTM model** (646K parameters) — it learned the shape of the language, but the replies barely made sense.
3. **This transformer** — clearly better than both. My early models also overfitted badly; dropout, weight decay and early stopping fixed most of that.

A real example from the trained model:

```
BOY: Kaisi ho?
GIRL: Mai bas itna bura lag rha tha
```

The replies are short and in Hinglish, because that is what the training data looks like.

## Run it

**Train / explore**

- `data_exploration.ipynb` — how the data was cleaned and checked
- `token_training.ipynb` — the tokenizer, all three models, training and test results

Install the packages in `requirements.txt` first. Training also needs the data files in `data/raw/`, which are not in this repo.

**Chat with it**

```
streamlit run deployment/app.py
```

The trained weights and tokenizer files are stored in a private Hugging Face repo. The app downloads them at startup, so it needs an `HF_TOKEN` in Streamlit secrets (or as an environment variable).

## Files

```
data_exploration.ipynb    data cleaning and exploration
token_training.ipynb      tokenizer + training (bigram, LSTM, transformer)
deployment/app.py         Streamlit chat app
deployment/model.py       the transformer model + reply generation
requirements.txt          packages needed for training
```

## What I learned

- Attention is not magic. The causal mask is the main trick that turns it into a language model.
- A model this small overfits fast. Half the work of training is stopping it from just memorising.
- The tokenizer matters more than I expected — the model only ever sees what its tokenizer gives it.
- A time-ordered test set is honest. A random split leaks future messages into training and makes the results look better than they are.

## Next

- Longer context — 64 tokens is short for a real conversation
- A bigger version of the model, compared properly on perplexity
- Try different sampling settings and note what actually changes
