# ANN — Bank Churn Prediction

This folder was earlier named "Loan Prediction" — that name was wrong. The notebook actually predicts **bank customer churn**: whether a customer will leave the bank or stay.

## What it does

- Dataset: the standard bank churn dataset — credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary
- Target: `Exited` — 1 if the customer left, 0 if they stayed
- Preprocessing: one-hot encoding for geography and gender, feature scaling
- Model: a Keras ANN (TensorFlow), trained for up to 100 epochs

## Result

- Test accuracy: **0.85** (2,000 test rows)
- Validation accuracy stays around 0.85 almost from the start — the model plateaus early, so most of the 100 epochs add nothing. Early stopping would cut training time without changing the result.

## Files

```
ANN_Churn_Prediction.ipynb    preprocessing, training and evaluation
```

## What I learned

- On tabular data, a neural network does not automatically beat simpler models — clean features and an honest test split matter more.
- Watch the validation curve. When it goes flat, more epochs are just more waiting.
