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

#Currency conversion (June 2026)
INR_TO_USD = 0.0106          # ₹1 ≈ $0.0106  (about ₹94 to $1)

def usd(inr):
    """Convert a rupee amount to a short USD string like '$221'."""
    return f"\\${inr * INR_TO_USD:,.0f}"

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
    avg = df[TARGET].mean()
    col3.metric("Average fare", f"₹{avg:,.0f} (${avg * INR_TO_USD:,.0f})")

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

    plot_df = df.sample(min(5000, len(df)), random_state=1)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Distribution 💵",
        "Price by Class 🎫",
        "Price vs Days Left 🕐",
        "Price by Airline ✈️",
        "Correlation 🔥",
    ])

    # Tab 1: Price distribution
    with tab1:
        st.markdown("#### How much do flights cost?")
        fig, ax = plt.subplots()
        sns.histplot(df[TARGET], bins=50, kde=True, color=FLIGHT_BLUE, ax=ax)
        ax.set_xlabel("Price (INR)")
        ax.set_title("Most fares cluster low, with a long tail of pricey flights")
        st.pyplot(fig)
        median_price = df[TARGET].median()
        st.caption(f"Half of all flights cost under ₹{median_price:,.0f} "
                   f"(about {usd(median_price)}). That long right tail is mostly "
                   "Business class, which we split out next.")

    # Tab 2: Price by class
    with tab2:
        st.markdown("#### Does cabin class drive the fare?")
        order = df.groupby("class")[TARGET].median().sort_values().index
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="class", y=TARGET, order=order,
                    color=FLIGHT_BLUE, showfliers=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Price (INR)")
        ax.set_title("Business class costs several times more than Economy")
        st.pyplot(fig)
        med = df.groupby("class")[TARGET].median()
        if {"Business", "Economy"}.issubset(med.index):
            ratio = med["Business"] / med["Economy"]
            st.caption(f"Median Business fare is ₹{med['Business']:,.0f} ({usd(med['Business'])}), "
                       f"about {ratio:.0f} times the median Economy fare of "
                       f"₹{med['Economy']:,.0f} ({usd(med['Economy'])}). "
                       "Class is the strongest single predictor of price in this data.")
        else:
            st.caption("Business fares sit dramatically higher than Economy.")

    # Tab 3: Price vs days_left
    with tab3:
        st.markdown("#### Do prices climb as departure approaches?")
        by_days = df.groupby("days_left")[TARGET].mean()
        fig, ax = plt.subplots()
        sns.lineplot(x=by_days.index, y=by_days.values, marker="o",
                     color=FLIGHT_GOLD, linewidth=2.5, ax=ax)
        ax.invert_xaxis()
        ax.set_xlabel("Days left until departure")
        ax.set_ylabel("Average price (INR)")
        ax.set_title("Fares spike in the final days before departure")
        st.pyplot(fig)
        last_min = by_days.loc[by_days.index <= 2].mean()
        early    = by_days.loc[by_days.index >= 40].mean()
        st.caption(f"Booking last-minute averages ₹{last_min:,.0f} ({usd(last_min)}) versus "
                   f"₹{early:,.0f} ({usd(early)}) booking 40+ days out. This is the dataset's "
                   "window into dynamic pricing, and why days_left carries real signal.")

    # Tab 4: Price by airline 
    with tab4:
        st.markdown("#### Which airlines charge the most?")
        by_air = df.groupby("airline")[TARGET].median().sort_values()
        fig, ax = plt.subplots()
        sns.barplot(x=by_air.values, y=by_air.index, color=FLIGHT_BLUE, ax=ax)
        ax.set_xlabel("Median price (INR)")
        ax.set_ylabel("")
        ax.set_title("Median fare by airline")
        st.pyplot(fig)
        priciest = by_air.index[-1]
        top_price = by_air.iloc[-1]
        st.caption(f"{priciest} has the highest median fare at ₹{top_price:,.0f} "
                   f"({usd(top_price)}), partly a mix effect: full-service carriers sell more "
                   "Business seats, which pulls their median up.")

    # Tab 5: Correlation 
    with tab5:
        st.markdown("#### Correlation between the numeric columns")
        corr = df[NUMERIC + [TARGET]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=BLUE_SEQ,
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title("How duration and days_left relate to price")
        st.pyplot(fig)
        corr_days = corr.loc["days_left", TARGET]
        sign = "negative" if corr_days < 0 else "positive"
        st.caption(f"days_left has a {sign} linear correlation ({corr_days:.2f}) with price. "
                   "The link is mostly flat until the final stretch, so one straight-line number "
                   "understates it, a good argument for also trying a tree-based model.")

    st.caption(f"USD figures use an approximate rate of ₹1 ≈ \\${INR_TO_USD} (June 2026).")

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
