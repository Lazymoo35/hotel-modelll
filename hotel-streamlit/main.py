import streamlit as st
import numpy as np
import pandas as pd
import joblib

# NOTE FOR NEXT PROGRESS: Real trial ended before load model (2:20:00), if error try after that step!

# Page Config
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="centered", page_icon="🏨")

# Load model
# model = PLACEHOLDER_FOR_MODEL_LOADING_CODE  # JANLUP GANTII!

# Get prediction
def predict(data:pd.DataFrame, model):
    """Get prediction from model
    
    Args:
        data (pd.DataFrame): dataframe
        model: classification model
    """
    prediction = model.predict(data)
    predict_proba = model.predict_proba(data)
    prediction_label = prediction.map({0: "Won't Cancel", 1: "Will Cancel"})
    return {
        "prediction": prediction,
        "proba": predict_proba,
        "label": prediction_label
    }

# Page texts
st.title("Hotel Booking Cancellation Prediction")
st.write("This app predicts whether a hotel booking will be cancelled based on user inputs.")
st.subheader("User Input Parameters")

# Creating the columns
col1, col2, col3 = st.columns(3, gap="medium", border=True)

with col1:
    hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=730, value=0)
    arrival_date_month = st.selectbox("Arrival Month", 
        ["January", "February", "March", "April", "May", "June", 
         "July", "August", "September", "October", "November", "December"])
    arrival_date_day_of_month = st.number_input("Arrival Day of Month", min_value=1, max_value=31, value=1)
    stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=30, value=0)
    stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=0)
    adults = st.number_input("Number of Adults", min_value=0, max_value=30, value=0)
    children = st.number_input("Number of Children", min_value=0, max_value=30, value=0)
    babies = st.number_input("Number of Babies", min_value=0, max_value=30, value=0)
    

with col2:
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=100, value=0)
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=100, value=0)
    booking_changes = st.number_input("Booking Changes", min_value=0, max_value=100, value=0)
    days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=365, value=0)
    adr = st.number_input("Average Daily Rate", min_value=0, max_value=1000, value=0)
    required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=10, value=0)
    total_of_special_requests = st.number_input("Total of Special Requests", min_value=0, max_value=10, value=0)

with col3:
    is_repeated_guest = st.checkbox("Is Repeated Guest")
    country = st.text_input("Country (ISO Code)")
    market_segment = st.selectbox("Market Segment", ["Direct", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Online TA/TO", "Corporate", "Aviation", "Undefined"])
    distribution_channel = st.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
    reserved_room_type = st.selectbox("Reserved Room Type", list("ABCDEFGHIJKL"))
    assigned_room_type = st.selectbox("Assigned Room Type", list("ABCDEFGHIJKL"))
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])
    meal = st.selectbox("Meal Plan", ["BB", "FB", "HB", "SC"])

# Predict Button
predict_button = st.button("Predict Cancellation", use_container_width=True)
if predict_button:
    df = pd.DataFrame(
        np.array([[hotel_type, lead_time, arrival_date_month, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
                   previous_cancellations, previous_bookings_not_canceled, booking_changes, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests,
                   is_repeated_guest, country, market_segment, distribution_channel, deposit_type, reserved_room_type, assigned_room_type, customer_type, meal]]),
        columns=["hotel", "lead_time", "arrival_date_month", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                 "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
                 "is_repeated_guest", "country", "market_segment", "distribution_channel", "deposit_type", "reserved_room_type", "assigned_room_type", "customer_type", "meal"]
    )
    st.write(df)

    # PRedict
    result = predict(df, model)
    st.write(f"Guest with this booking {label}, with the probability being {proba}.")




    # f"Guest with this booking {prediction}, with the probability being {predict_proba}."