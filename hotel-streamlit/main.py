import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# CUSTOM TRANSFORMER

class ColumnCombiner(BaseEstimator, TransformerMixin):
    """
    Combine multiple numeric columns into a single new column.
    Example: adults + children + babies -> total_guests
    """
    def __init__(self, columns=None, new_column_name=None):
        self.columns = columns or []
        self.new_column_name = new_column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if all(col in X.columns for col in self.columns):
            X[self.new_column_name] = X[self.columns].fillna(0).sum(axis=1)
            X.drop(self.columns, axis=1, inplace=True, errors="ignore")
        return X
    
    def get_feature_names_out(self):
        return [self.new_column_name]
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drop specified columns if they exist in the DataFrame.
    """
    def __init__(self, columns=None):
        self.columns = columns or []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors="ignore")
    
class MultiColumnCombiner(BaseEstimator, TransformerMixin):
    """
    Combine multiple sets of columns into new features at once.
    Example:
    combiner = MultiColumnCombiner(
        combinations={
            "total_guests": ["adults", "children", "babies"],
            "total_stay_nights": ["stays_in_weekend_nights", "stays_in_week_nights"]
        }
    )
    """
    def __init__(self, combinations=None):
        self.combinations = combinations or {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for new_col, cols in self.combinations.items():
            if all(c in X.columns for c in cols):
                X[new_col] = X[cols].fillna(0).sum(axis=1)
                X.drop(cols, axis=1, inplace=True, errors="ignore")
        return X
    
    def get_feature_names_out(self):
        return list(self.combinations.keys())

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, whisker=1.5):
        self.columns = columns
        self.whisker = whisker
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                s = pd.to_numeric(X[col], errors="coerce").dropna()
                if len(s) > 0:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    self.bounds_[col] = (q1 - self.whisker*iqr, q3 + self.whisker*iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lb, ub) in self.bounds_.items():
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].clip(lower=lb, upper=ub).fillna(X[col].median())
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.columns

class ADRScaler(BaseEstimator, TransformerMixin):
    """Scale adr column using RobustScaler"""
    def __init__(self, column="adr"):
        self.column = column
        self.scaler = RobustScaler()
        self.fitted_ = False
    
    def fit(self, X, y=None):
        if self.column in X.columns:
            self.scaler.fit(X[[self.column]])
            self.fitted_ = True
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.fitted_ and self.column in X.columns:
            X[self.column] = self.scaler.transform(X[[self.column]])
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.columns

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.01, n_categories=1, replace_with="Other"):
        self.tol = tol
        self.n_categories = n_categories
        self.replace_with = replace_with

    def fit(self, X, y=None):
        # pastikan DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.columns_ = X.columns
        self.rare_categories_ = {}
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            rare_cats = freqs[freqs < self.tol].index
            if len(freqs) - len(rare_cats) < self.n_categories:
                rare_cats = freqs.sort_values().index[self.n_categories:]
            self.rare_categories_[col] = set(rare_cats)
        return self

    def transform(self, X):
        # pastikan DataFrame
        array_input = False
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
            array_input = True

        X_copy = X.copy()
        for col, rare_cats in self.rare_categories_.items():
            X_copy[col] = X_copy[col].where(~X_copy[col].isin(rare_cats), self.replace_with)

        return X_copy.values if array_input else X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.columns_

# CUSTOM TRANSFORMER

# START OF APP

# Page Config
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="centered", page_icon="🏨")

# Load model
model = joblib.load("Hotel_Cancellation.joblib")

# Get prediction
def predict(data:pd.DataFrame, models):
    """Get prediction from model
    
    Args:
        data (pd.DataFrame): dataframe
        model: classification model
    """
    prediction = model.predict(data)
    predict_proba = model.predict_proba(data)
    prediction_label = pd.Series(prediction).map({0: "won't cancel", 1: "will cancel"})
    return {
        "prediction": prediction,
        "proba": predict_proba,
        "label": prediction_label
    }

# Page texts
st.title("Hotel Booking Cancellation Prediction")
st.write("This app predicts whether a hotel booking will be cancelled based on user inputs.")
st.subheader("User Input Parameters")

tab1, tab2 = st.tabs(["Input and Prediction", "Projected Cost"])

# Header options
hotel_type = st.radio("Hotel Type", ["City Hotel", "Resort Hotel"])
country = st.text_input("Country (ISO Code)")

# Creating the columns
col1, col2, col3 = st.columns(3, gap="medium", border=True)

with col1:
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=730, value=0)
    arrival_date_month = st.selectbox("Arrival Month", 
        ["January", "February", "March", "April", "May", "June", 
         "July", "August", "September", "October", "November", "December"])
    arrival_date_day_of_month = st.slider("Arrival Day of Month", min_value=1, max_value=31, value=1)  
    
with col2:
    days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=365, value=0)
    stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=30, value=0)
    stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, value=0)
    
with col3:
    adults = st.number_input("Number of Adults", min_value=0, max_value=30, value=0)
    children = st.number_input("Number of Children", min_value=0, max_value=30, value=0)
    babies = st.number_input("Number of Babies", min_value=0, max_value=30, value=0)
    
    
first_expander_trigger = (stays_in_weekend_nights + stays_in_week_nights) > 0

with st.expander("FInancial Info", expanded=first_expander_trigger):
    adr = st.number_input("Average Daily Rate", min_value=0, max_value=1000, value=0)
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
    market_segment = st.selectbox("Market Segment", ["Direct", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Online TA/TO", "Corporate", "Aviation", "Undefined"])
    distribution_channel = st.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])

second_expander_trigger = adr != 0

with st.expander("Guests' Request", expanded=second_expander_trigger):
    reserved_room_type = st.selectbox("Reserved Room Type", list("ABCDEFGHIJKL"))
    meal = st.selectbox("Meal Plan", ["BB", "FB", "HB", "SC", "Undefined"])
    required_car_parking_spaces = st.slider("Required Car Parking Spaces", min_value=0, max_value=10, value=0)
    total_of_special_requests = st.slider("Total of Special Requests", min_value=0, max_value=10, value=0)
    booking_changes = st.slider("Booking Changes", min_value=0, max_value=20, value=0)

is_repeated_guest = st.checkbox("Is Repeated Guest")
with st.expander("Only fill if repeated guest!", expanded=is_repeated_guest):
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=100, value=0)
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=100, value=0)


# Predict Button
predict_button = st.button("Predict Cancellation", use_container_width=True)
if predict_button:
    df = pd.DataFrame(
        np.array([[hotel_type, lead_time, arrival_date_month, arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
                   previous_cancellations, previous_bookings_not_canceled, booking_changes, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests,
                   is_repeated_guest, country, market_segment, distribution_channel, deposit_type, reserved_room_type, customer_type, meal]]),
        columns=["hotel", "lead_time", "arrival_date_month", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                 "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
                 "is_repeated_guest", "country", "market_segment", "distribution_channel", "deposit_type", "reserved_room_type", "customer_type", "meal"]
    )
    st.write(df)

    # PRedict
    result = predict(df, model)
    st.write(f"Guest with this booking **{result['label'].iloc[0]}**, with the probability being **{result['proba'][0].max():.2f}**.")