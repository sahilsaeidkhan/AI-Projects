from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==================================================
# Load model and artifacts
# ==================================================

model = joblib.load(
    MODEL_DIR / "flight_delay_xgb.pkl"
)

threshold = joblib.load(
    MODEL_DIR / "threshold.pkl"
)

default_delay_rate = joblib.load(
    MODEL_DIR / "default_delay_rate.pkl"
)


# ==================================================
# FastAPI
# ==================================================

app = FastAPI(
    title="Flight Delay Predictor API",
    description="Predicts whether a flight will arrive 15+ minutes late.",
    version="1.0.0"
)


# ==================================================
# Input schema
# ==================================================

class FlightInput(BaseModel):

    day_of_month: int
    day_of_week: int
    week_of_month: int

    airline: str

    origin_city: str
    origin_state: str

    destination_city: str
    destination_state: str

    departure_hour: int
    departure_minute: int

    arrival_hour: int
    arrival_minute: int

    elapsed_time: float
    distance: float


# ==================================================
# Root endpoint
# ==================================================

@app.get("/")
def root():

    return {
        "message": "Flight Delay Predictor API is running",
        "threshold": threshold
    }


# ==================================================
# Prediction endpoint
# ==================================================

@app.post("/predict")
def predict(flight: FlightInput):

    # ----------------------------------------------
    # Route features
    # ----------------------------------------------

    route = (
        flight.origin_city
        + " -> "
        + flight.destination_city
    )

    airline_route = (
        flight.airline
        + "_"
        + route
    )


    # ----------------------------------------------
    # Time features
    # ----------------------------------------------

    departure_minutes = (
        flight.departure_hour * 60
        + flight.departure_minute
    )

    departure_time_sin = np.sin(
        2 * np.pi * departure_minutes / 1440
    )

    departure_time_cos = np.cos(
        2 * np.pi * departure_minutes / 1440
    )


    # ----------------------------------------------
    # Day-of-week cyclic features
    # ----------------------------------------------

    dow_sin = np.sin(
        2 * np.pi * flight.day_of_week / 7
    )

    dow_cos = np.cos(
        2 * np.pi * flight.day_of_week / 7
    )


    # ----------------------------------------------
    # Historical features
    #
    # Prototype fallback:
    # use overall training delay rate.
    #
    # Later we can replace these with actual
    # date-aware historical lookup tables.
    # ----------------------------------------------

    carrier_historical_delay_rate = default_delay_rate

    origin_historical_delay_rate = default_delay_rate

    destination_historical_delay_rate = default_delay_rate

    route_historical_delay_rate = default_delay_rate

    departure_hour_historical_delay_rate = default_delay_rate


    # ----------------------------------------------
    # Build model input
    # ----------------------------------------------

    input_data = pd.DataFrame([
        {
            "DAY_OF_MONTH": flight.day_of_month,
            "DAY_OF_WEEK": flight.day_of_week,
            "WEEK_OF_MONTH": flight.week_of_month,

            "OP_UNIQUE_CARRIER": flight.airline,

            "ORIGIN_CITY_NAME": flight.origin_city,
            "ORIGIN_STATE_ABR": flight.origin_state,

            "DEST_CITY_NAME": flight.destination_city,
            "DEST_STATE_ABR": flight.destination_state,

            "ROUTE": route,
            "AIRLINE_ROUTE": airline_route,

            "DEP_HOUR": flight.departure_hour,
            "DEP_MINUTE": flight.departure_minute,

            "ARR_HOUR": flight.arrival_hour,
            "ARR_MINUTE": flight.arrival_minute,

            "DEP_TIME_SIN": departure_time_sin,
            "DEP_TIME_COS": departure_time_cos,

            "DOW_SIN": dow_sin,
            "DOW_COS": dow_cos,

            "CRS_ELAPSED_TIME": flight.elapsed_time,
            "DISTANCE": flight.distance,

            "carrier_historical_delay_rate":
                carrier_historical_delay_rate,

            "origin_historical_delay_rate":
                origin_historical_delay_rate,

            "destination_historical_delay_rate":
                destination_historical_delay_rate,

            "route_historical_delay_rate":
                route_historical_delay_rate,

            "departure_hour_historical_delay_rate":
                departure_hour_historical_delay_rate
        }
    ])


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= threshold
    )


    # ----------------------------------------------
    # Risk level
    # ----------------------------------------------

    if probability < 0.20:

        risk_level = "LOW"

    elif probability < 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # ----------------------------------------------
    # Response
    # ----------------------------------------------

    return {

        "delay_probability":
            round(float(probability), 4),

        "delay_probability_percent":
            round(float(probability * 100), 2),

        "prediction":
            "DELAYED" if prediction == 1
            else "NOT DELAYED",

        "risk_level":
            risk_level,

        "threshold":
            threshold
    }