import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Flight Price Predictor ✈️",
    layout="centered",
    page_icon="✈️",
)

FLIGHT_BLUE = "#2E5EAA"  
FLIGHT_GOLD = "#F2A900"  
FLIGHT_DARK = "#0E1B2A"  
BLUE_SEQ    = "Blues"    

sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"]   = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"]   = 11
plt.rcParams["axes.edgecolor"]   = "#cccccc"
plt.rcParams["figure.autolayout"] = True   

## Step 01 - Sidebar navigation
st.sidebar.title("Flight Price – Fare Analysis ✈️")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Business Case 📘",
        "Data Description 🗂️",
        "Visualization 📊",
        "Prediction 🔮",
        "Explainable AI 🔍",
        "Hyperparameter Tuning 📈",
        "Conclusion ✅",
    ],
)

## Step 02 - Load the data
DATA_PATH = "flights.csv"  

CATEGORICAL = ["airline", "source_city", "departure_time", "stops",
               "arrival_time", "destination_city", "class"]
NUMERIC = ["duration", "days_left"]
TARGET  = "price"   # prices are in Indian Rupees (EaseMyTrip data)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])  # leftover index
    df = df.drop(columns=[c for c in ["flight"] if c in df.columns])      # high-cardinality ID
    return df

df = load_data(DATA_PATH)

## Page 01 — Business Case (the landing page)
if page == "Business Case 📘":
    st.title("Flight Price Prediction ✈️")
    st.markdown("#### Can we predict what a flight should cost, and tell travelers when to book?")

    st.markdown(
        """
**The problem.** Flight prices feel random to travelers, but they aren't. Airlines use
**dynamic pricing**: fares shift with demand, how many seats are left, the time of day,
the route, and how far ahead you book. That makes it hard for a traveler to know whether
a price they're seeing is fair, or whether they should book now or wait.

**Our goal.** Build a model that estimates a fair fare from a flight's attributes, so a
traveler-facing tool could answer two questions:
1. Is this price reasonable for this route and timing?
2. Should I book now, or are prices likely to climb as the date nears?

**Who it helps.** Travelers comparing options, and on the other side, an airline or travel
platform stress-testing its own pricing.
        """
    )

    # Headline metrics for the speech
    c1, c2, c3 = st.columns(3)
    c1.metric("Flights in dataset", f"{df.shape[0]:,}")
    c2.metric("Routes (cities)", f"{df['source_city'].nunique()} → {df['destination_city'].nunique()}")
    c3.metric("Average fare", f"₹{df[TARGET].mean():,.0f}")

    st.info(
        "Prices are in Indian Rupees (₹). The data comes from EaseMyTrip, "
        "covering flights between India's top metro cities."
    )

## Page 02 — Data Description
elif page == "Data Description 🗂️":
    st.title("Data Description 🗂️")
    st.markdown(
        "Each **row is one flight option** that was available to book. The columns describe "
        "the flight (who flies it, the route, the timing, the class) and how far ahead it was "
        "priced. The variable we predict is **price**."
    )

    # Shape at a glance
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Target", "price (₹)")

    # Data dictionary — what each column means
    st.markdown("##### What each column means")
    col_info = pd.DataFrame({
        "Column": ["airline", "source_city", "departure_time", "stops",
                   "arrival_time", "destination_city", "class",
                   "duration", "days_left", "price"],
        "Type": ["Categorical", "Categorical", "Categorical", "Categorical",
                 "Categorical", "Categorical", "Categorical",
                 "Numerical", "Numerical", "Numerical (target)"],
        "Description": [
            "Airline operating the flight",
            "City the flight departs from",
            "Time-of-day bucket for departure (e.g. Morning, Evening)",
            "Number of stops (zero, one, two_or_more)",
            "Time-of-day bucket for arrival",
            "City the flight lands in",
            "Seat class (Economy or Business)",
            "Total travel time in hours",
            "Days between booking and departure",
            "Ticket price in Indian Rupees — what we predict",
        ],
    })
    st.dataframe(col_info, hide_index=True, use_container_width=True)

    # Live preview
    st.markdown("##### Data preview")
    rows = st.slider("Rows to display", 5, 50, 5)
    st.dataframe(df.head(rows), use_container_width=True)

    # Missing values check
    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found ✅")
    else:
        st.warning("⚠️ Some columns have missing values:")
        st.dataframe(missing[missing > 0])

    # Categorical vs numerical breakdown
    st.markdown("##### Feature types")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Categorical**")
        st.write(CATEGORICAL)
    with c2:
        st.markdown("**Numerical**")
        st.write(NUMERIC + [TARGET])

    # Summary statistics (tucked in an expander to keep the page clean)
    with st.expander("Show summary statistics"):
        st.dataframe(df.describe(), use_container_width=True)

elif page == "Visualization 📊":
    st.title("Visualization 📊")
    st.info("🚧 Building this next.")

elif page == "Prediction 🔮":
    st.title("Prediction 🔮")
    st.info("🚧 Building this next.")

elif page == "Explainable AI 🔍":
    st.title("Explainable AI 🔍")
    st.info("🚧 Building this next.")

elif page == "Hyperparameter Tuning 📈":
    st.title("Hyperparameter Tuning 📈")
    st.info("🚧 Building this next.")

elif page == "Conclusion ✅":
    st.title("Conclusion ✅")
    st.info("🚧 Building this next.")
