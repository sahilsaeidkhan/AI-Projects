# Music Generation with an LSTM

This is a character-level LSTM that learns music in ABC notation and then writes new tunes in the same format.

The songs are in `data/`. I converted the text into numbers, one ID per character, and trained the model to predict the next character. The LSTM has a hidden size of 256 and is built in PyTorch. Trained weights are in `lstm_music_model.pth`, and one sample it generated is in `generated_music.abc`.

## How the runs went

First run, 15 epochs: loss came down from 4.40 to 0.97.

Second run: validation accuracy 56.43%. But look at the losses. Training loss keeps going down (0.73 to 0.65) while validation loss goes up (1.56 to 1.62). That is overfitting. The model should have been saved at its best validation point, not at the end. That fix is still pending.

Judging this model only by its loss would be a mistake. The real test is reading the generated music and checking if it follows the ABC format. It mostly does, and that is more satisfying than the loss curve.

## Files

```
music_generation.ipynb    data prep, training and generation
data/                     ABC-notation songs used for training
generated_music.abc       a sample the model generated
lstm_music_model.pth      trained model weights
```
