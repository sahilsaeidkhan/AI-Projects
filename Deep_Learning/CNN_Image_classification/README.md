# CNN Image Classification (MNIST)

I trained two models on MNIST handwritten digits in this notebook.

The first one is a simple fully connected network. Test accuracy: 93.77%. After that I built a CNN on the same data. Test accuracy: 98.77%.

That difference is the whole point. A dense network treats every pixel as a separate number. A CNN looks at small patches and learns shapes like edges and curves, which is how digits are actually formed.

## Files

```
CNN_Mnist_Image_Classification.ipynb    both models, training and evaluation (PyTorch)
```

## Notes

- The CNN ran for 20 epochs. Its final average loss was around 0.021.
- There is an early run in the notebook where the loss stays stuck near 2.3. I have kept it in. The fixed runs are below it.
- I have not saved the trained CNN model yet. The other projects in this repo save theirs, so this one still needs that cleanup.
- The notebook ends with the CNN's wrong predictions. Looking at those taught me more than the accuracy number did.
