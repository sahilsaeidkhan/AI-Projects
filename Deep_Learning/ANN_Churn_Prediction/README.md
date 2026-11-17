# ANN Churn Prediction

This folder used to be called "Loan Prediction". That name was wrong. The notebook predicts bank customer churn, meaning whether a customer will leave the bank or stay.

## About the notebook

- Dataset: the usual bank churn dataset. Features are credit score, geography, gender, age, tenure, balance, number of products, credit card status, active member and estimated salary.
- Target is `Exited`. 1 means the customer left, 0 means they stayed.
- Geography and gender are one-hot encoded, and the numeric features are scaled.
- The model is a Keras ANN, set to train for 100 epochs.

## Result

Test accuracy is 0.85 on 2,000 test rows. Validation accuracy reaches about 0.85 very early and then stays flat, so most of the 100 epochs are wasted. Early stopping is the obvious fix. I just have not applied it yet.

Two things I took from this project. On tabular data, a neural network is not automatically better than a simple model. And if the validation curve is flat, adding more epochs only adds waiting time.

## Files

```
ANN_Churn_Prediction.ipynb    preprocessing, training and evaluation
```
