## Step 00 - Import packages
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

# Shared color scheme
FLIGHT_BLUE = "#2E5EAA"   # primary (airline blue)
FLIGHT_GOLD = "#F2A900"   # accent  (livery gold)
FLIGHT_DARK = "#0E1B2A"   # night-sky navy
BLUE_SEQ    = "Blues"     # for the heatmap

# Make every matplotlib/seaborn chart use the same look
sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["figure.autolayout"] = True   # stops labels getting cut off

## Step 01 - Sidebar navigation
st.sidebar.title("Flight Price – Fare Analysis ✈️")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Introduction 📘",
        "Visualization 📊",
        "Prediction 🔮",
        "Explainable AI 🔍",
        "Hyperparameter Tuning 📈",
        "Conclusion ✅",
    ],
)
# st.image("header.png")   # uncomment once you upload an image

## Step 02 - Load and prepare the dataset
CATEGORICAL = ["airline", "source_city", "departure_time", "stops",
               "arrival_time", "destination_city", "class"]
NUMERIC = ["duration", "days_left"]
TARGET  = "price"   # prices are in Indian Rupees (EaseMyTrip data)

@st.cache_data
def load_data(path="flights.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "flight"], errors="ignore")  # index + ID col
    df = df.dropna()
    return df

df = load_data()

## Step 03 - Pages
if page == "Introduction 📘":
    st.title("✈️ Predicting Flight Prices in India")
    st.subheader("01 Introduction 📘")
    st.caption(f"EaseMyTrip bookings across {df['source_city'].nunique()} metro cities "
               f"and {df['airline'].nunique()} airlines. Prices are in Indian Rupees (₹).")

    st.markdown("**Goal:** predict the price of a flight from its airline, route, timing, "
                "stops, duration, and how far ahead it's booked.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Average fare", f"₹{df[TARGET].mean():,.0f}")

    st.markdown("##### Data Preview")
    view = st.radio("Show from:", ["Head (top)", "Tail (bottom)"], horizontal=True)
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    if view == "Head (top)":
        st.dataframe(df.head(rows))
    else:
        st.dataframe(df.tail(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("No missing values found")
    else:
        st.warning("You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    st.dataframe(df.describe())

elif page == "Visualization 📊":
    st.subheader("02 Data Visualization 📊")
    st.info("")

elif page == "Prediction 🔮":
    st.subheader("03 Prediction 🔮")
    st.info("")

elif page == "Explainable AI 🔍":
    st.subheader("04 Explainable AI 🔍")
    st.info("")

elif page == "Hyperparameter Tuning 📈":
    st.subheader("05 Hyperparameter Tuning 📈")
    st.info("")

elif page == "Conclusion ✅":
    st.subheader("06 Conclusion ✅")
    st.info("")
