import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌿",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
    background-color:#F7FAF5;
}

section[data-testid="stSidebar"]{
    background:#EAF4E5;
}

h1{
    color:#1B5E20;
}

h2,h3{
    color:#2E7D32;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:12px;
    padding:18px;
    border:1px solid #DCE9D8;
    box-shadow:0px 2px 8px rgba(0,0,0,0.05);
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_PATH = "data/cleaned_aqi_dataset.csv"

try:
    df = pd.read_csv(DATA_PATH)
    data_loaded = True
except Exception:
    data_loaded = False

    # ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = "models/xgboost_aqi_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌿 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualizations",
        "🤖 Prediction",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model : Random Forest & XGBoost")

st.sidebar.markdown("---")

st.sidebar.caption("AQI Prediction System")

# ==========================================================
# HOME PAGE
# ==========================================================

if page=="🏠 Home":

    st.title("🌍 AQI Prediction System")

    st.subheader("Machine Learning Based Air Quality Intelligence Dashboard")

    st.markdown(
        """
Predict • Analyze • Visualize • Protect
"""
    )

    st.divider()

    if data_loaded:

        rows = df.shape[0]
        cols = df.shape[1]

        avg_aqi = round(df["AQI"].mean(),2)

        max_aqi = round(df["AQI"].max(),2)

    else:

        rows=0
        cols=0
        avg_aqi=0
        max_aqi=0

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric(
            "Records",
            rows
        )

    with c2:
        st.metric(
            "Features",
            cols
        )

    with c3:
        st.metric(
            "Average AQI",
            avg_aqi
        )

    with c4:
        st.metric(
            "Maximum AQI",
            max_aqi
        )

    st.divider()

    if data_loaded:

        st.success("Dataset Loaded Successfully ✅")

    else:

        st.error("Dataset Not Found")

    st.markdown("## About Project")

    st.write("""
This project predicts Air Quality Index (AQI)
using Machine Learning.

The prediction model has been trained using
Prayagraj Air Quality Dataset.

The objective of this dashboard is to help users
understand air pollution levels through data
analysis and intelligent prediction.
""")

    st.divider()

    st.info("""
🌿 Features

• Dataset Overview

• Interactive Visualizations

• AQI Prediction

• Health Awareness

• Machine Learning Model
""")

    st.divider()

    st.caption("Version 1.0")

# ==========================================================
# DATASET PAGE
# ==========================================================

elif page=="📊 Dataset":

    st.title("📊 Dataset")

    if data_loaded:

        st.success("Dataset Loaded")

        st.dataframe(df)

    else:

        st.error("Dataset Not Found")

# ==========================================================
# VISUALIZATION PAGE
# ==========================================================

elif page=="📈 Visualizations":

    st.title("📈 AQI Data Visualizations")

    if not data_loaded:
        st.error("Dataset Not Found")
        st.stop()

    st.success("Interactive Dashboard Loaded")

    # ===============================
    # Select Location
    # ===============================

    locations = ["All Locations"] + sorted(df["Location"].unique().tolist())

    selected_location = st.selectbox(
        "Select Location",
        locations
    )

    if selected_location != "All Locations":
        filtered_df = df[df["Location"] == selected_location]
    else:
        filtered_df = df.copy()

    st.divider()

    # ===============================
    # AQI Distribution
    # ===============================

    st.subheader("AQI Distribution")

    fig = px.histogram(
        filtered_df,
        x="AQI",
        nbins=30,
        color_discrete_sequence=["green"],
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # AQI by Location
    # ===============================

    st.subheader("Average AQI by Location")

    avg_location = (
        df.groupby("Location")["AQI"]
        .mean()
        .reset_index()
        .sort_values("AQI", ascending=False)
    )

    fig = px.bar(
        avg_location,
        x="Location",
        y="AQI",
        color="AQI",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Correlation Heatmap
    # ===============================

    st.subheader("Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(include="number")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Scatter Plot
    # ===============================

    st.subheader("AQI vs Pollutant")

    pollutants = [
        "PM2.5 (µg/m³)",
        "PM10 (µg/m³)",
        "NO2 (µg/m³)",
        "SO2 (µg/m³)",
        "NH3 (µg/m³)",
        "O3 (µg/m³)"
    ]

    selected_pollutant = st.selectbox(
        "Choose Pollutant",
        pollutants
    )

    fig = px.scatter(
        filtered_df,
        x=selected_pollutant,
        y="AQI",
        color="AQI",
        size="AQI",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Box Plot
    # ===============================

    st.subheader("AQI Spread")

    fig = px.box(
        filtered_df,
        y="AQI",
        color_discrete_sequence=["darkgreen"],
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Pollutant Comparison
    # ===============================

    st.subheader("Average Pollutant Levels")

    pollutant_avg = pd.DataFrame({
        "Pollutant":[
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "NH3",
            "O3"
        ],
        "Average":[
            filtered_df["PM2.5 (µg/m³)"].mean(),
            filtered_df["PM10 (µg/m³)"].mean(),
            filtered_df["NO2 (µg/m³)"].mean(),
            filtered_df["SO2 (µg/m³)"].mean(),
            filtered_df["NH3 (µg/m³)"].mean(),
            filtered_df["O3 (µg/m³)"].mean()
        ]
    })

    fig = px.bar(
        pollutant_avg,
        x="Pollutant",
        y="Average",
        color="Average",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.success("Visualization Dashboard Completed ✅")

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page=="🤖 Prediction":

    st.title("🤖 AQI Prediction")

    st.info("Coming in Part 3")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page=="ℹ About":

    st.title("About")

    st.write("AQI Prediction System")

    st.write("Machine Learning Model : XGBoost")

    st.write("Framework : Streamlit")

    st.write("Language : Python")

    st.write("Dataset : Prayagraj Air Quality Dataset")