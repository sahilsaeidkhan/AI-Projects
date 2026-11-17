# Loan Prediction System

This project works in two stages.

Stage 1 is a Random Forest classifier. It looks at a loan application and predicts approved or rejected. Stage 2 is a Random Forest regressor. It runs only when stage 1 says approved, and it predicts the loan amount.

## Results

From `analyze.ipynb`:

- Stage 1 accuracy: 0.98 (OOB score 0.986)
- Stage 2 is trained only on approved loans. Its detailed metrics are in the notebook.

One thing worth knowing: CIBIL score alone has about 84% of the feature importance in stage 1. So in this dataset, approval is mostly decided by CIBIL. The other features barely move the result.

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

I built it in two stages because one model doing both jobs was messy. Splitting the decision and the amount made each part simpler to train and simpler to check. The accuracy taught me something too: a 0.98 score can still hide a very simple model, so check the feature importance before trusting a big number.
