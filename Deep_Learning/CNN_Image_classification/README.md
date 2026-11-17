# CNN — MNIST Image Classification

Two models on the MNIST handwritten digits dataset, in one notebook:

1. A simple fully connected network first — **test accuracy 93.77%**
2. Then a CNN — **test accuracy 98.77%**

That jump is the point of the project. For images, convolution beats plain dense layers, because the model learns local patterns (edges, curves) instead of treating every pixel as an independent number.

## Files

```
CNN_Mnist_Image_Classification.ipynb    both models, training and evaluation (PyTorch)
```

## Notes

- The CNN was trained for 20 epochs; its average loss came down to about 0.021.
- An early broken run (loss stuck around 2.3) is still in the notebook — the fixed runs come after it.
- The trained model is not saved in this folder. Saving it (like the other projects do) is a pending cleanup.

## What I learned

- Architecture matters more than effort. The same data and similar training gave +5% accuracy just by switching to convolution.
- Always check what the model gets wrong — the notebook ends with the CNN's wrong predictions, which teaches more than the accuracy number.
