import streamlit as st
import requests
from datetime import date, time, datetime



API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)



st.title("✈️ Flight Delay Predictor")
st.write(
    "Predict whether a flight is likely to arrive "
    "15+ minutes late."
)

st.divider()



st.subheader("Flight Information")

col1, col2 = st.columns(2)

with col1:
    airline = st.text_input(
        "Airline Code",
        value="AA",
        help="Example: AA, DL, UA, WN"
    )

    origin_city = st.text_input(
        "Origin City",
        value="New York, NY"
    )

    origin_state = st.text_input(
        "Origin State",
        value="NY"
    )

with col2:
    destination_city = st.text_input(
        "Destination City",
        value="Los Angeles, CA"
    )

    destination_state = st.text_input(
        "Destination State",
        value="CA"
    )

    flight_date = st.date_input(
        "Flight Date",
        value=date.today()
    )



st.subheader("Schedule")

col1, col2 = st.columns(2)

with col1:
    departure_time = st.time_input(
        "Scheduled Departure",
        value=time(18, 30)
    )

with col2:
    arrival_time = st.time_input(
        "Scheduled Arrival",
        value=time(21, 45)
    )

col1, col2 = st.columns(2)

with col1:
    elapsed_time = st.number_input(
        "Scheduled Elapsed Time (minutes)",
        min_value=1,
        max_value=1500,
        value=375
    )

with col2:
    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        max_value=10000,
        value=2475
    )



st.subheader("Historical Delay Information")

st.info(
    "These values are currently required by the trained model. "
    "For this prototype, enter estimated historical delay rates "
    "between 0 and 1."
)

col1, col2 = st.columns(2)

with col1:
    carrier_rate = st.number_input(
        "Carrier Historical Delay Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.21,
        step=0.01
    )

    origin_rate = st.number_input(
        "Origin Historical Delay Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.18,
        step=0.01
    )

    destination_rate = st.number_input(
        "Destination Historical Delay Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.23,
        step=0.01
    )

with col2:
    route_rate = st.number_input(
        "Route Historical Delay Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01
    )

    departure_hour_rate = st.number_input(
        "Departure Hour Historical Delay Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )



st.divider()

predict_button = st.button(
    "🔮 Predict Flight Delay",
    type="primary",
    use_container_width=True
)

if predict_button:



    day_of_month = flight_date.day

    day_of_week = flight_date.weekday()

    week_of_month = ((day_of_month - 1) // 7) + 1



    departure_hour = departure_time.hour
    departure_minute = departure_time.minute

    arrival_hour = arrival_time.hour
    arrival_minute = arrival_time.minute


    payload = {
        "day_of_month": day_of_month,
        "day_of_week": day_of_week,
        "week_of_month": week_of_month,

        "airline": airline,
        "origin_city": origin_city,
        "origin_state": origin_state,

        "destination_city": destination_city,
        "destination_state": destination_state,

        "departure_hour": departure_hour,
        "departure_minute": departure_minute,

        "arrival_hour": arrival_hour,
        "arrival_minute": arrival_minute,

        "elapsed_time": elapsed_time,
        "distance": distance,

        "carrier_historical_delay_rate": carrier_rate,
        "origin_historical_delay_rate": origin_rate,
        "destination_historical_delay_rate": destination_rate,
        "route_historical_delay_rate": route_rate,
        "departure_hour_historical_delay_rate": departure_hour_rate
    }

    # ----------------------------------------------
    # Send request to FastAPI
    # ----------------------------------------------

    try:

        with st.spinner("Predicting..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            probability = result["delay_probability_percent"]
            prediction = result["prediction"]
            risk = result["risk_level"]
            threshold = result["threshold"]

            st.divider()

            st.subheader("Prediction Result")

       

            st.metric(
                "Delay Probability",
                f"{probability:.2f}%"
            )

       

            if prediction == "DELAYED":
                st.error(
                    f"⚠️ Prediction: {prediction}"
                )
            else:
                st.success(
                    f"✅ Prediction: {prediction}"
                )

   
            st.write(
                f"**Risk Level:** {risk}"
            )

            st.caption(
                f"Model decision threshold: {threshold:.2f}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure the API is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out."
        )

    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )