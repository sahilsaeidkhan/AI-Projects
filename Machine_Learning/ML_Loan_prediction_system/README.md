# Loan Prediction System — a two-stage model

This project predicts two things about a loan application:

1. Will the loan be approved? (classification)
2. If yes, what amount does the model expect? (regression)

It is a two-stage Random Forest system. Stage 1 decides approval. Stage 2 runs only for approved applications and predicts the amount.

## Results (from analyze.ipynb)

| Stage | Model | Score |
| --- | --- | --- |
| 1 — approval | Random Forest classifier | Accuracy 0.98 · OOB score 0.986 |
| 2 — amount | Random Forest regressor | Trained only on approved loans (details in analyze.ipynb) |

One honest note: CIBIL score alone carries about 84% of the feature importance in stage 1. In this dataset, approval is mostly a CIBIL decision — the other features change very little.

## Files

```
analyze.ipynb    data analysis, training and evaluation
loan_model.py    loads both saved pipelines and runs the two-stage prediction
app.py           Streamlit app
api.py           FastAPI app (POST /predict)
models/          stage_1_rf_classifier_pipeline.pkl, stage_2_rf_regression_pipeline.pkl
data/            loan_approval_dataset.csv
```

## Run it

Install the packages first:

```
pip install -r requirements.txt
```

Then run from inside this folder:

```
streamlit run app.py
```

or the API:

```
uvicorn api:app --reload
```

## What I learned

- Splitting a problem into two stages — first decide, then estimate — is cleaner than forcing one model to do both.
- A 0.98 accuracy can still hide a simple model. Check feature importance before celebrating.
