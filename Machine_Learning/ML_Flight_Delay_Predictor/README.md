# ✈️ Flight Delay Predictor

An end-to-end machine learning application that predicts whether a
flight is likely to arrive 15+ minutes late.

## Architecture

Streamlit Frontend
        ↓
FastAPI
        ↓
XGBoost Model
        ↓
Prediction

## Features

The model uses:

- Airline
- Origin and destination
- Flight date
- Departure time
- Arrival time
- Scheduled elapsed time
- Distance
- Historical delay features

## Machine Learning

Models explored:

- Dummy Classifier
- Logistic Regression
- Random Forest
- XGBoost

The final model is an XGBoost classifier.

## Evaluation

The project uses a chronological train/validation/test split
to reduce temporal leakage.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Joblib

## Running Locally

### Install dependencies

```bash
pip install -r requirements.txt