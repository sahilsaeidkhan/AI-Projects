# RNN — Music Generation

A character-level LSTM that learns music written in ABC notation, then generates new tunes in the same format.

## How it works

- Data: ABC-notation songs in `data/`
- The text is converted to numbers (one ID per character), and the model learns to predict the next character
- Model: LSTM with hidden size 256 (PyTorch)
- Trained weights are saved in `lstm_music_model.pth`, and one generated sample is in `generated_music.abc`

## Results

- First run (15 epochs): loss came down from 4.40 to 0.97
- Second run: validation accuracy **56.43%** — but validation loss rises (1.56 → 1.62) while training loss keeps falling. That is overfitting. The fix is to keep the best checkpoint instead of the final one.

## Files

```
music_generation.ipynb    data prep, training and generation
data/                     ABC-notation songs used for training
generated_music.abc       a sample the model generated
lstm_music_model.pth      trained model weights
```

## What I learned

- A generative model can be judged two ways — the loss curve, and actually listening to (or reading) what it creates. The second one matters more.
- When validation loss turns upward, training longer is making the model worse, not better.
