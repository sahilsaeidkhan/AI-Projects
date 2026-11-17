# ✈️ Flight Delay Predictor

An end-to-end machine learning application that predicts whether a flight is likely to arrive 15+ minutes late.

## How it works

```
Streamlit UI  →  FastAPI  →  XGBoost model  →  prediction + risk level
```

- `UI/app.py` — the form the user fills
- `api/main.py` — builds the model features and returns the delay probability, the prediction (DELAYED / NOT DELAYED) and a risk level (LOW / MEDIUM / HIGH)
- `models/` — the saved model and the decision threshold (0.20)
- `data_audit.ipynb` — data audit, feature engineering, training and evaluation

## Features the model uses

- Airline, origin and destination (city, state, route)
- Date features — day of month, day of week (also as sin/cos), week of month
- Departure and arrival time (also as sin/cos)
- Scheduled elapsed time and distance
- Historical delay rates — for now these fall back to the overall training delay rate (0.2133); date-aware lookup tables are a planned improvement

## Models tried

Validation results. The split is chronological — the model is always tested on later flights than it trained on.

| Model | ROC-AUC | F1 |
| --- | --- | --- |
| Dummy classifier | — | 0.00 |
| Logistic Regression | 0.560 | 0.147 |
| Random Forest | 0.601 | 0.251 |
| XGBoost (tuned) | 0.618 | — |
| XGBoost (class-weighted, final) | 0.619 | 0.306 |

## Final test results

The saved model is the class-weighted XGBoost, used with a threshold of 0.20 instead of the usual 0.50.

| Metric | Value |
| --- | --- |
| Recall | 0.943 |
| Precision | 0.111 |
| F1 | 0.199 |
| ROC-AUC | 0.537 |
| PR-AUC | 0.127 |

Read this honestly: the low threshold catches 94% of delayed flights, but most of its warnings are false alarms, and the test ROC-AUC (0.54) is lower than validation (0.62). Flight delay prediction from schedule data alone is a hard problem — weather and live flight status are not in this data. Accuracy at this threshold is 0.157, which is exactly why accuracy is the wrong number to judge this model by.

## Run it locally

Install the packages:

```bash
pip install -r requirements.txt
```

Start the API (from this folder):

```bash
uvicorn api.main:app --reload
```

Then start the UI in a second terminal:

```bash
streamlit run UI/app.py
```

The UI sends requests to the API at `http://127.0.0.1:8000/predict`.

## Data

The dataset is US flight on-time data (BTS reporting). The raw CSV is about 73 MB, so it is not stored in this repo. To retrain, download it and place it in `data/` — the notebook expects `data/T_ONTIME_REPORTING.csv`.

## What I learned

- With time-based data, split by time, not randomly — a random split leaks the future into training.
- On imbalanced data, accuracy lies. A dummy model that never predicts a delay scores 0.84 accuracy and is completely useless.
- The threshold is a decision, not a default. 0.50 is rarely right when the two kinds of mistakes cost differently.
