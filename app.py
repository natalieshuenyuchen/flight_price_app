import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

st.set_page_config(page_title="Flight Price Prediction ✈️", layout="wide", page_icon="✈️")

st.sidebar.title("Flight Price Prediction ✈️")
page = st.sidebar.radio(
    "Go to page",
    [
        "01 Business Case 📘",
        "02 Visualization 📊",
        "03 Prediction 🤖",
        "04 Explainable AI 🔍",
        "05 Hyperparameter Tuning 📈",
        "06 Conclusion ✅",
    ],
)

DATA_PATH = "flights.csv"
CATEGORICAL = ["airline", "source_city", "departure_time", "stops",
               "arrival_time", "destination_city", "class"]
NUMERIC = ["duration", "days_left"]
TARGET = "price"

