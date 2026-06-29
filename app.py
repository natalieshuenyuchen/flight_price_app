## Step 00 - Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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
st.image("header.png")   

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

## Train the two models once (cached)
@st.cache_resource(show_spinner="Training the two models (runs once)...")
def train_models(_df):
    work = _df
    if len(work) > 50000:                      # keeps training fast + inside Streamlit's memory
        work = work.sample(50000, random_state=42)

    X = work[CATEGORICAL + NUMERIC]
    y = work[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def make_preprocessor():
        # named "cat" / "num" so the SHAP page can find them later
        return ColumnTransformer(transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])

    # Pipeline steps named "preprocessor" / "model" so the SHAP page can unpack them
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)),
        ]),
    }

    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "MAE":  mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R²":   r2_score(y_test, preds),
        })

    results_df = pd.DataFrame(rows).set_index("Model")
    best_model_name = results_df["MAE"].idxmin()
    return models, results_df, best_model_name, X_test, y_test


models, results_df, best_model_name, X_test, y_test = train_models(df)

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

    st.markdown("""
    This page compares two regression models for predicting flight prices:

    - **Linear Regression**: an interpretable baseline model
    - **Random Forest Regressor**: a nonlinear tree-based model that can capture complex pricing patterns

    The goal is not only to predict fares, but also to understand which model performs better for this tabular dataset.
    """)

    st.markdown("### Model Performance Comparison")

    display_results = results_df.copy()
    display_results["MAE"]  = display_results["MAE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
    display_results["RMSE"] = display_results["RMSE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
    display_results["R²"]   = display_results["R²"].map(lambda x: f"{x:.3f}")
    st.dataframe(display_results, use_container_width=True)

    st.success(f"Best model based on lowest MAE: **{best_model_name}**")

    st.markdown("### Try Your Own Flight Prediction")

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", sorted(df["airline"].unique()))
        source_city = st.selectbox("Source City", sorted(df["source_city"].unique()))
        departure_time = st.selectbox("Departure Time", sorted(df["departure_time"].unique()))
        stops = st.selectbox("Stops", sorted(df["stops"].unique()))

    with col2:
        arrival_time = st.selectbox("Arrival Time", sorted(df["arrival_time"].unique()))

        destination_options = sorted([
            city for city in df["destination_city"].unique()
            if city != source_city
        ])
        destination_city = st.selectbox("Destination City", destination_options)

        flight_class = st.selectbox("Class", sorted(df["class"].unique()))

        duration = st.slider(
            "Duration in hours",
            float(df["duration"].min()),
            float(df["duration"].max()),
            float(df["duration"].median())
        )

        days_left = st.slider(
            "Days left before departure",
            int(df["days_left"].min()),
            int(df["days_left"].max()),
            int(df["days_left"].median())
        )

    input_df = pd.DataFrame([{
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": flight_class,
        "duration": duration,
        "days_left": days_left,
    }])

    selected_model_name = st.selectbox(
        "Choose a model for prediction",
        list(models.keys())
    )

    selected_model = models[selected_model_name]
    predicted_price = selected_model.predict(input_df)[0]

    st.metric(
        label=f"Predicted price using {selected_model_name}",
        value=f"₹{predicted_price:,.0f}"
    )

    st.caption(f"Approximately {usd(predicted_price)}")

    st.caption(
        "This prediction is based on historical flight listings. It should be interpreted as an estimated fare benchmark, not a guaranteed ticket price."
    )

elif page == "Explainable AI 🔍":
    st.subheader("04 Explainable AI 🔍")

    st.markdown("""
    Explainable AI helps us understand **why** the model predicts a certain fare.
    In this project, we use **SHAP (SHapley Additive exPlanations)** to interpret the Random Forest model.

    SHAP shows how each feature pushes a prediction higher or lower compared with the model's average prediction.
    """)

    rf_model = models["Random Forest Regressor"]

    st.markdown("### Global Feature Importance with SHAP")

    # Extract preprocessor and trained Random Forest model
    preprocessor = rf_model.named_steps["preprocessor"]
    rf = rf_model.named_steps["model"]

    # Use a user-selected sample for SHAP so the app runs quickly
    shap_n = st.slider(
        "Choose SHAP sample size",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Larger samples give more stable explanations but make the app slower."
    )

    shap_sample = X_test.sample(min(shap_n, len(X_test)), random_state=42)

    st.caption(
        f"SHAP values are computed on a representative sample of {len(shap_sample):,} test rows "
        "to keep the app responsive."
    )

    # Transform data using the fitted preprocessor
    shap_X = preprocessor.transform(shap_sample)

    # Convert sparse matrix to dense if needed
    if hasattr(shap_X, "toarray"):
        shap_X = shap_X.toarray()

    # Get transformed feature names
    numeric_features = NUMERIC
    categorical_features = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    all_features = np.concatenate([numeric_features, categorical_features])

    # Calculate SHAP values
    with st.spinner("Calculating SHAP values... this may take a moment."):
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(shap_X)

    # SHAP summary bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        shap_X,
        feature_names=all_features,
        plot_type="bar",
        show=False,
        max_display=15
    )
    st.pyplot(fig)

    st.markdown("""
    The SHAP bar chart ranks the most important features by their average impact on the model's predictions.
    Features with larger SHAP values have stronger influence on predicted flight prices.
    """)

    st.markdown("### Local Explanation for One Flight")

    example_idx = st.slider(
        "Choose a test example",
        min_value=0,
        max_value=min(len(shap_sample) - 1, 199),
        value=0,
        step=1
    )

    example_original = shap_sample.iloc[[example_idx]]
    example_transformed = shap_X[[example_idx]]

    actual_price = y_test.loc[example_original.index[0]]
    predicted_price = rf_model.predict(example_original)[0]

    col1, col2 = st.columns(2)

    col1.metric("Actual Price", f"₹{actual_price:,.0f}")
    col1.caption(f"Approximately {usd(actual_price)}")

    col2.metric("Predicted Price", f"₹{predicted_price:,.0f}")
    col2.caption(f"Approximately {usd(predicted_price)}")

    st.markdown("##### Flight details")
    st.dataframe(example_original)

    st.markdown("##### SHAP Waterfall Plot")

    st.caption(
        "Note: numeric values in the waterfall plot are standardized by the preprocessing pipeline, "
        "while categorical variables are shown as one-hot encoded indicators."
    )

    expected_value = explainer.expected_value

    # For regression, expected_value is usually a scalar
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]

    shap_explanation = shap.Explanation(
        values=shap_values[example_idx],
        base_values=expected_value,
        data=example_transformed[0],
        feature_names=all_features
    )

    fig2 = plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap_explanation,
        max_display=8,
        show=False
    )
    
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("""
    The waterfall plot explains one individual prediction. Features shown in red push the predicted fare higher,
    while features shown in blue push it lower. This helps users understand why one flight is predicted to be
    more expensive or less expensive than the model's average prediction.
    """)

    st.markdown("### Business Interpretation")

    st.write("""
    SHAP makes the model more transparent for travelers and booking platforms. Instead of only showing a predicted fare,
    the app can explain whether that fare is mainly driven by cabin class, airline, route, flight duration, number of stops,
    or how close the booking is to the departure date.
    """)

elif page == "Hyperparameter Tuning 📈":
    st.subheader("05 Hyperparameter Tuning 📈")
    st.write("Search Random Forest settings, log every run to Weights & Biases, and pick the best.")

    # W&B is optional so the page still works for testing without a key
    use_wandb = st.checkbox("Log runs to Weights & Biases", value=False)
    wandb_project, wandb_entity, wandb_key = "flight-price-tuning", "", ""
    if use_wandb:
        c1, c2, c3 = st.columns(3)
        wandb_key     = c1.text_input("W&B API key", type="password",
                                      help="From wandb.ai/authorize")
        wandb_project = c2.text_input("Project name", "flight-price-tuning")
        wandb_entity  = c3.text_input("Entity (username)", "")

    st.markdown("### Search space")
    c1, c2 = st.columns(2)
    n_list = c1.multiselect("n_estimators", [50, 100, 200, 300], default=[50, 100])
    d_list = c2.multiselect("max_depth", [10, 20, 30], default=[10, 20])
    sample_n = st.slider("Training sample size (smaller = faster)",
                         5000, 50000, 20000, step=5000)

    if st.button("🚀 Run sweep"):
        # Build the same X/y the rest of the app uses
        work = df.sample(min(sample_n, len(df)), random_state=42)
        X = work[CATEGORICAL + NUMERIC]
        y = work[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        if use_wandb and wandb_key:
            import wandb
            wandb.login(key=wandb_key)

        combos = [(n, d) for n in n_list for d in d_list]
        if not combos:
            st.warning("Pick at least one value for each parameter.")
            st.stop()

        results = []
        bar = st.progress(0.0, text="Training combinations...")
        for i, (n, d) in enumerate(combos):
            pipe = Pipeline([
                ("preprocessor", ColumnTransformer([
                    ("num", StandardScaler(), NUMERIC),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ])),
                ("model", RandomForestRegressor(
                    n_estimators=n, max_depth=d, random_state=42, n_jobs=-1)),
            ])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            mae  = mean_absolute_error(y_te, preds)
            r2   = r2_score(y_te, preds)
            results.append({"n_estimators": n, "max_depth": d,
                            "RMSE": rmse, "MAE": mae, "R²": r2})

            if use_wandb and wandb_key:
                run = wandb.init(project=wandb_project,
                                 entity=wandb_entity or None,
                                 name=f"rf_n{n}_d{d}", reinit=True,
                                 config={"n_estimators": n, "max_depth": d})
                wandb.log({"RMSE": rmse, "MAE": mae, "R2": r2})
                run.finish()

            bar.progress((i + 1) / len(combos), text=f"Trained {i+1}/{len(combos)}")

        st.session_state["tuning_results"] = pd.DataFrame(results).sort_values("RMSE")

    # Show results (persists after the button run)
    if "tuning_results" in st.session_state:
        res = st.session_state["tuning_results"].reset_index(drop=True)
        st.markdown("### Results (sorted by RMSE, lower is better)")

        show = res.copy()
        show["RMSE"] = show["RMSE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
        show["MAE"]  = show["MAE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
        show["R²"]   = show["R²"].map(lambda x: f"{x:.3f}")
        st.dataframe(show, use_container_width=True)

        best = res.iloc[0]
        st.success(
            f"Best: n_estimators={int(best['n_estimators'])}, "
            f"max_depth={int(best['max_depth'])} → "
            f"RMSE ₹{best['RMSE']:,.0f} ({usd(best['RMSE'])}), R² {best['R²']:.3f}"
        )

        labels = res.apply(lambda r: f"n{int(r['n_estimators'])}_d{int(r['max_depth'])}", axis=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, res["RMSE"], color=FLIGHT_BLUE)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Hyperparameter comparison")
        st.pyplot(fig)

        if use_wandb:
            st.info(f"Live dashboard: wandb.ai → project '{wandb_project}'")
    
elif page == "Conclusion ✅":
    st.subheader("06 Conclusion ✅")

    # Pull the winning model and score it on the held-out test set
    best_model = models[best_model_name]
    preds = best_model.predict(X_test)
    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    st.markdown(f"### Best model: **{best_model_name}**")
    st.write(
        "We compared an interpretable Linear Regression baseline against a Random Forest "
        "that can capture nonlinear pricing patterns. Here's how the winner performs on "
        "flights it never saw during training."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("R² (variance explained)", f"{r2:.3f}")
    c2.metric("Average error (MAE)", f"₹{mae:,.0f} ({usd(mae)})")
    c3.metric("Typical error (RMSE)", f"₹{rmse:,.0f} ({usd(rmse)})")
    st.caption(
        f"On average the prediction lands within about {usd(mae)} of the real fare, "
        f"and the model explains roughly {r2*100:.0f}% of why fares differ."
    )

    st.markdown("### How the two models compared")
    show = results_df.copy()
    show["MAE"]  = show["MAE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
    show["RMSE"] = show["RMSE"].map(lambda x: f"₹{x:,.0f} (${x*INR_TO_USD:,.0f})")
    show["R²"]   = show["R²"].map(lambda x: f"{x:.3f}")
    st.dataframe(show, use_container_width=True)
    st.caption(
        "Random Forest wins because flight pricing is nonlinear: the jump near departure "
        "and the gap between Economy and Business aren't straight lines, which a linear "
        "model can't bend to fit."
    )

    st.markdown("### Predicted vs actual fares")
    sample = pd.DataFrame({"Actual": y_test.values, "Predicted": preds}) \
                .sample(min(2000, len(y_test)), random_state=1)
    fig, ax = plt.subplots()
    ax.scatter(sample["Actual"], sample["Predicted"], alpha=0.3, color=FLIGHT_BLUE)
    lims = [sample["Actual"].min(), sample["Actual"].max()]
    ax.plot(lims, lims, "--", color=FLIGHT_GOLD, linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual fare (INR)")
    ax.set_ylabel("Predicted fare (INR)")
    ax.set_title("Points near the gold line are good predictions")
    ax.legend()
    st.pyplot(fig)
    st.caption("The model tracks fares well across the range, with the most spread on the "
               "priciest Business tickets, where fewer examples exist to learn from.")

    st.markdown("### What we learned")
    st.markdown(
        """
- **Cabin class is the biggest driver of price.** Business fares run several times higher
  than Economy, and the SHAP analysis confirms class dominates every other feature.
- **Booking timing is real and learnable.** Fares stay flat for weeks, then spike in the
  final days before departure. That dynamic-pricing pattern is exactly why days_left
  carries signal, and why the nonlinear model outperforms the linear one.
- **The model is honest about its limits.** An R² below 1.0 is expected. Real airline
  pricing also reacts to the exact travel date, live seat inventory, and competitor fares,
  none of which are in this dataset. The gap is the part of pricing the data can't see.
        """
    )

    st.markdown("### Business value")
    st.write(
        "Even with those limits, the model is accurate enough to power a fair-price check "
        "or a book-now-versus-wait nudge for travelers, and the SHAP explanations make it "
        "transparent enough to trust. The natural next step is adding travel dates and "
        "seasonality so it can see holiday demand."
    )
