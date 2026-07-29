import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
import base64

st.markdown(
    """
<link rel="stylesheet" 
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""",
    unsafe_allow_html=True,
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(page_title="AQI Prediction System", page_icon="🌿", layout="wide")

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go To", ["🏠 Home", "📊 Dataset", "📈 Visualizations", "🤖 Prediction", "ℹ About"]
)

st.sidebar.markdown("---")

st.sidebar.success("Model : Random Forest & XGBoost")

st.sidebar.markdown("---")

st.sidebar.caption("AQI Prediction System")

# ==========================================================
# SIDEBAR EXTRA INFORMATION
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.info(
"""
🌍 **Project Focus**

Air Quality Monitoring

AI Based AQI Prediction

Data Driven Insights
"""
)


st.sidebar.success(
"""
🤖 **Technology Stack**

🐍 Python

📊 Data Analytics

🧠 Machine Learning

⚡ XGBoost Model
"""
)


st.sidebar.markdown("---")


st.sidebar.markdown(
"""
<div style="
background:#e8f5e9;
padding:15px;
border-radius:12px;
text-align:center;
">

<h4 style="color:#2e7d32;">
🌱 Smart Environment
</h4>

<p style="font-size:13px;">
Predict Air Quality<br>
Protect Health<br>
Build Awareness
</p>

</div>
""",
unsafe_allow_html=True
)


st.sidebar.markdown("---")

st.sidebar.caption(
"💚 AI for Cleaner Tomorrow"
)

# ==========================================================
# CHAPTER 1 : HERO SECTION
# ==========================================================

if page == "🏠 Home":

    hero_image = Image.open("Images/Prayagraj_Hero_image.png")

    st.image(hero_image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # CHAPTER 2 : WHY THIS PLATFORM?
    # ==========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <h2 style="
    text-align:center;
    color:#2e7d32;
    font-size:34px;
    margin-bottom:30px;
    ">
    🌿 Why This Platform?
    </h2>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
    ### 📂 Real Dataset

    ✔ Prayagraj Air Quality Dataset

    ✔ Cleaned & Preprocessed

    ✔ Reliable Environmental Data
    """)

    with c2:
        st.info("""
    ### 🤖 AI Powered

    ✔ Random Forest Model

    ✔ XGBoost Model

    ✔ Intelligent AQI Prediction
    """)

    with c3:
        st.warning("""
    ### 📊 Interactive Dashboard

    ✔ Visual Analytics

    ✔ AQI Prediction

    ✔ User-Friendly Interface
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # CHAPTER 4 : WHY AQI MATTERS
    # ==========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <h2 style="
    text-align:center;
    color:#2e7d32;
    font-size:34px;
    margin-bottom:25px;
    ">
    🌍 Why AQI Matters?
    </h2>
    """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("""
    Air pollution is one of the world's biggest environmental challenges.

    Monitoring the **Air Quality Index (AQI)** helps citizens understand pollution levels and take precautions before health conditions become serious.

    This dashboard combines **Machine Learning**, **Data Analytics**, and **Interactive Visualization** to provide meaningful AQI insights for Prayagraj.
    """)

    with right:

        st.success("🌿 **Good (0–50)**\n\nFresh and healthy air.")

        st.info("🟡 **Moderate (51–100)**\n\nAcceptable for most people.")

        st.warning("🟠 **Poor (101–200)**\n\nSensitive groups should be careful.")

        st.error("🔴 **Unhealthy (201+)**\n\nHealth risks increase significantly.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # CHAPTER 5 : DID YOU KNOW?
    # ==========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <h2 style="
    text-align:center;
    color:#2e7d32;
    font-size:34px;
    margin-bottom:30px;
    ">
    💡 Did You Know?
    </h2>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
    🌿 **Fact 1**

    Air pollution is responsible for millions of premature deaths worldwide every year.
    """)

        st.success("""
    🍃 **Fact 2**

    An AQI below **50** indicates clean and healthy air for everyone.
    """)

    with col2:

        st.warning("""
    🌫 **Fact 3**

    Machine Learning can identify pollution patterns and help predict future AQI levels.
    """)

        st.error("""
    🚨 **Fact 4**

    People with asthma and heart disease are more vulnerable when AQI exceeds **200**.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # CHAPTER 6 : ABOUT THIS SYSTEM
    # ==========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <div style="
    background:linear-gradient(135deg,#ffffff,#f8fff8);
    padding:28px;
    border-radius:20px;
    border:1px solid #dcedc8;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    ">

    <h2 style="
    color:#2e7d32;
    margin-bottom:15px;
    ">
    📘 About This System
    </h2>

    <p style="
    font-size:17px;
    color:#555;
    line-height:1.8;
    text-align:justify;
    ">

    This AI-powered dashboard analyzes the
    Prayagraj Air Quality Dataset to predict the
    Air Quality Index (AQI) using Machine Learning.

    The platform combines intelligent prediction,
    interactive visualization and environmental
    awareness to help users better understand
    air pollution trends and make informed decisions.

    </p>

    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # 📊 PREMIUM PROJECT STATISTICS
    # ==========================================================

    st.markdown(
        """
        <h2 style="
        text-align:center;
        color:#2e7d32;
        margin-top:20px;
        margin-bottom:25px;
        ">
        📊 Project Statistics
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Calculate Statistics
    # ----------------------------

    if data_loaded:

        total_records = df.shape[0]
        total_features = df.shape[1]
        average_aqi = round(df["AQI"].mean(), 2)
        maximum_aqi = round(df["AQI"].max(), 2)

    else:

        total_records = 0
        total_features = 0
        average_aqi = 0
        maximum_aqi = 0

    # ----------------------------
    # Statistics Cards
    # ----------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(label="📄 Dataset Records", value=f"{total_records:,}")

    with col2:

        st.metric(label="🧪 Prediction Inputs", value="9")

    with col3:

        st.metric(label="🤖 ML Models", value="2")

    with col4:

        st.metric(label="📈 Average AQI", value=average_aqi)

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # More Details
    # ----------------------------

    with st.expander("🔍 View Complete Project Statistics"):

        st.markdown(f"""
    ### 📊 Dataset Summary

    - 📄 **Total Records:** {total_records:,}
    - 📑 **Total Columns:** {total_features}
    - 🧪 **Prediction Inputs:** 9
    - 🤖 **Machine Learning Models:** 2
    - 📈 **Average AQI:** {average_aqi}
    - 🚨 **Maximum AQI:** {maximum_aqi}

    ---

    ### 🌿 Prediction Input Parameters

    - PM2.5
    - PM10
    - NO₂
    - SO₂
    - NH₃
    - O₃
    - Temperature
    - Humidity
    - Wind Speed

    ---

    ### 🤖 Models Used

    ✅ Random Forest Regressor

    ✅ XGBoost Regressor

    ---

    ### 🎯 Purpose

    This AI-powered dashboard analyzes environmental parameters and predicts the Air Quality Index (AQI) for better environmental awareness and decision making.

    """)

        # ==========================================================
    # PROFESSIONAL FOOTER
    # ==========================================================

    from datetime import datetime

    st.divider()

    st.markdown(
        """
        <div style="
        background:#f8fff8;
        border:1px solid #dcedc8;
        border-radius:18px;
        padding:28px;
        text-align:center;
        box-shadow:0px 4px 15px rgba(0,0,0,0.06);
        ">

        <h3 style="
        color:#2e7d32;
        margin-bottom:8px;
        ">
        🌿 AQI Prediction System
        </h3>

        <p style="
        color:#555;
        font-size:16px;
        margin-bottom:18px;
        ">
        AI-Powered Air Quality Prediction Platform for
        Prayagraj, Uttar Pradesh
        </p>

        <hr style="
        border:0.5px solid #dcedc8;
        ">

        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Footer Information
    # -----------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 👩‍💻 Developer")
        st.write("**Anjali Tiwari**")
        st.caption("B.Tech CSE (Data Science)")

    with c2:
        st.markdown("### 🤖 Technology")
        st.write("Random Forest")
        st.write("XGBoost")
        st.caption("Machine Learning")

    with c3:
        st.markdown("### 📅 Project Info")
        st.write("**Version : 1.0**")
        st.write(f"Last Updated : {datetime.now().strftime('%d %B %Y')}")
        st.caption("Streamlit Dashboard")

    st.markdown("---")

    st.markdown(
        """
        <div style="
        text-align:center;
        color:#666;
        font-size:15px;
        line-height:1.8;
        ">

        ❤️ Built with <b>Python</b>, <b>Streamlit</b>,
        <b>Machine Learning</b> & <b>Data Analytics</b>

        <br><br>

        © 2026 <b>Anjali Tiwari</b> • All Rights Reserved

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset")

    if data_loaded:

        st.success("Dataset Loaded")

        st.dataframe(df)

    else:

        st.error("Dataset Not Found")

    # ==========================================================
    # CHAPTER 8 : DATASET STATUS
    # ==========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
    <h2 style="
    text-align:center;
    color:#2e7d32;
    font-size:32px;
    margin-bottom:25px;
    ">
    📂 Dataset Status
    </h2>
    """,
        unsafe_allow_html=True,
    )

    if data_loaded:

        st.markdown(
            """
        <div style="
        background:linear-gradient(135deg,#f1f8e9,#ffffff);
        border:1px solid #c8e6c9;
        border-radius:18px;
        padding:22px;
        text-align:center;
        box-shadow:0px 5px 15px rgba(0,0,0,0.08);
        ">

        <h3 style="color:#2e7d32;">
        ✅ Dataset Successfully Loaded
        </h3>

        <p style="
        color:#555;
        font-size:17px;
        line-height:1.7;
        ">

        Prayagraj Air Quality Dataset is ready for
        visualization, analysis and AQI prediction.

        </p>

        </div>
        """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
        <div style="
        background:#fff8f8;
        border:1px solid #ffcdd2;
        border-radius:18px;
        padding:22px;
        text-align:center;
        box-shadow:0px 5px 15px rgba(0,0,0,0.08);
        ">

        <h3 style="color:#c62828;">
        ❌ Dataset Not Found
        </h3>

        <p style="
        color:#666;
        font-size:17px;
        ">

        Please check the dataset file path and reload
        the application.

        </p>

        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# VISUALIZATION PAGE
# ==========================================================

elif page == "📈 Visualizations":

    st.title("📈 AQI Data Visualizations")

    if not data_loaded:
        st.error("Dataset Not Found")
        st.stop()

    st.success("Interactive Dashboard Loaded")

    # ===============================
    # Select Location
    # ===============================

    locations = ["All Locations"] + sorted(df["Location"].unique().tolist())

    selected_location = st.selectbox("Select Location", locations)

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
        template="plotly_white",
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
        avg_location, x="Location", y="AQI", color="AQI", template="plotly_white"
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
        corr, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto"
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
        "O3 (µg/m³)",
    ]

    selected_pollutant = st.selectbox("Choose Pollutant", pollutants)

    fig = px.scatter(
        filtered_df,
        x=selected_pollutant,
        y="AQI",
        color="AQI",
        size="AQI",
        template="plotly_white",
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
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ===============================
    # Pollutant Comparison
    # ===============================

    st.subheader("Average Pollutant Levels")

    pollutant_avg = pd.DataFrame(
        {
            "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "NH3", "O3"],
            "Average": [
                filtered_df["PM2.5 (µg/m³)"].mean(),
                filtered_df["PM10 (µg/m³)"].mean(),
                filtered_df["NO2 (µg/m³)"].mean(),
                filtered_df["SO2 (µg/m³)"].mean(),
                filtered_df["NH3 (µg/m³)"].mean(),
                filtered_df["O3 (µg/m³)"].mean(),
            ],
        }
    )

    fig = px.bar(
        pollutant_avg,
        x="Pollutant",
        y="Average",
        color="Average",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.success("Visualization Dashboard Completed ✅")

# ==========================================================
# PREDICTION PAGE
# ==========================================================


elif page == "🤖 Prediction":

    st.title("🤖 AQI Prediction")

    if not model_loaded:
        st.error("XGBoost Model Not Found")
        st.stop()

    st.success("Prediction Model Loaded Successfully")

    st.divider()

    # ==========================================================
    # ==========================================================
    # QUICK DEMO MODE
    # ==========================================================

    st.subheader("Enter Air Quality Parameters")

    st.info(
        "🧪 **Quick Demo Mode**\n\n"
        "Select a pre-defined air quality scenario to automatically fill realistic input values for demonstration and testing."
    )

    st.markdown("**Available Sample Scenarios**")

    scenario = st.selectbox(
        "",
        [
            "✍️ Custom Manual Input",
            "🟠 Moderate Air (Typical City Day)",
            "🔴 Poor Air (High Pollution)",
            "🟣 High Pollution (Near Dataset Maximum)",
            "🟢 Good Air (Demo Only)",
            "🟡 Satisfactory Air (Demo Only)",
            "⚫ Severe Air (Demo Only)",
        ],
    )

    st.caption(
        "💡 Note: Some sample scenarios are for demonstration only because they are outside the model's training data."
    )

    # ==========================================================
    # SAMPLE VALUES
    # ==========================================================

    samples = {
        "✍️ Custom Manual Input": dict(
            pm25=0,
            pm10=0,
            no2=0,
            so2=0,
            nh3=0,
            o3=0,
            temperature=0,
            humidity=0,
            wind_speed=0,
        ),
        "🟢 Good Air (Demo Only)": dict(
            pm25=25,
            pm10=45,
            no2=18,
            so2=6,
            nh3=18,
            o3=18,
            temperature=24,
            humidity=72,
            wind_speed=2,
        ),
        "🟡 Satisfactory Air (Demo Only)": dict(
            pm25=45,
            pm10=80,
            no2=25,
            so2=7,
            nh3=22,
            o3=20,
            temperature=25,
            humidity=73,
            wind_speed=2,
        ),
        "🟠 Moderate Air (Typical City Day)": dict(
            pm25=100,
            pm10=200,
            no2=35,
            so2=10,
            nh3=30,
            o3=20,
            temperature=25,
            humidity=70,
            wind_speed=2,
        ),
        "🔴 Poor Air (High Pollution)": dict(
            pm25=145,
            pm10=275,
            no2=42,
            so2=11,
            nh3=36,
            o3=26,
            temperature=26,
            humidity=75,
            wind_speed=1,
        ),
        "🟣 High Pollution (Near Dataset Maximum)": dict(
            pm25=158,
            pm10=300,
            no2=44,
            so2=11,
            nh3=39,
            o3=29,
            temperature=26,
            humidity=80,
            wind_speed=1,
        ),
        "⚫ Severe Air (Demo Only)": dict(
            pm25=220,
            pm10=420,
            no2=60,
            so2=18,
            nh3=50,
            o3=40,
            temperature=30,
            humidity=82,
            wind_speed=1,
        ),
    }
    values = samples[scenario]

    if scenario in [
        "🟠 Moderate Air (Typical City Day)",
        "🔴 Poor Air (High Pollution)",
        "🟣 High Pollution (Near Dataset Maximum)",
    ]:
        st.success(f"✅ '{scenario}' sample loaded successfully.")

    elif scenario in [
        "🟢 Good Air (Demo Only)",
        "🟡 Satisfactory Air (Demo Only)",
        "⚫ Severe Air (Demo Only)",
    ]:
        st.warning(
            "⚠️ This scenario is outside the model's training dataset.\n\n"
            "The model was trained on AQI values approximately between 120 and 275.\n"
            "Predictions for this scenario may not match the selected AQI category."
        )
    col1, col2 = st.columns(2)

    with col1:

        pm25 = st.number_input(
            "PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=float(values["pm25"]),
            step=1.0,
        )

        pm10 = st.number_input(
            "PM10 (µg/m³)",
            min_value=0.0,
            max_value=600.0,
            value=float(values["pm10"]),
            step=1.0,
        )

        no2 = st.number_input(
            "NO2 (µg/m³)",
            min_value=0.0,
            max_value=300.0,
            value=float(values["no2"]),
            step=1.0,
        )

        so2 = st.number_input(
            "SO2 (µg/m³)",
            min_value=0.0,
            max_value=200.0,
            value=float(values["so2"]),
            step=1.0,
        )

        nh3 = st.number_input(
            "NH3 (µg/m³)",
            min_value=0.0,
            max_value=200.0,
            value=float(values["nh3"]),
            step=1.0,
        )

    with col2:

        o3 = st.number_input(
            "O3 (µg/m³)",
            min_value=0.0,
            max_value=300.0,
            value=float(values["o3"]),
            step=1.0,
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=float(values["temperature"]),
            step=1.0,
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(values["humidity"]),
            step=1.0,
        )

        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=float(values["wind_speed"]),
            step=1.0,
        )

    st.divider()

    if st.button("🔮 Predict AQI", use_container_width=True):

        input_data = np.array(
            [[pm25, pm10, no2, so2, nh3, o3, temperature, humidity, wind_speed]]
        )

        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)

        # AQI Category

        if prediction <= 50:
            category = "🟢 Good"
            color = "green"
            advice = [
                "Air quality is excellent.",
                "Outdoor activities are completely safe.",
                "No mask is generally required.",
                "Ideal for walking and exercise.",
                "Keep monitoring AQI regularly.",
            ]

        elif prediction <= 100:
            category = "🟡 Satisfactory"
            color = "#8BC34A"
            advice = [
                "Air quality is acceptable.",
                "Outdoor activities are generally safe.",
                "Sensitive individuals should stay alert.",
                "Drink enough water.",
                "Continue monitoring pollution levels.",
            ]

        elif prediction <= 200:
            category = "🟠 Moderate"
            color = "orange"
            advice = [
                "Sensitive individuals should reduce prolonged outdoor exposure.",
                "Avoid heavy outdoor exercise.",
                "Children and elderly should take extra care.",
                "Wear a mask if pollution increases.",
                "Keep windows closed during peak pollution.",
            ]

        elif prediction <= 300:
            category = "🔴 Poor"
            color = "red"
            advice = [
                "Limit outdoor activities.",
                "Wear an N95 mask while outside.",
                "People with asthma should be cautious.",
                "Children and elderly should stay indoors.",
                "Use air purifiers if available.",
            ]

        elif prediction <= 400:
            category = "🟣 Very Poor"
            color = "purple"
            advice = [
                "Avoid outdoor activities.",
                "Wear a certified N95 mask.",
                "Keep doors and windows closed.",
                "Use indoor air purification.",
                "Consult a doctor if breathing discomfort occurs.",
            ]

        else:
            category = "⚫ Severe"
            color = "#5D1049"
            advice = [
                "Stay indoors as much as possible.",
                "Avoid all outdoor exercise.",
                "Wear an N95 mask if going outside is unavoidable.",
                "Children, elderly and patients should remain indoors.",
                "Follow local pollution advisories.",
            ]

            st.success("✅ Prediction Completed Successfully")

        st.markdown("## 🌍 AQI Status")

        st.markdown(
            f"""
                <div style="
                background:{color};
                padding:25px;
                border-radius:15px;
                color:white;
                text-align:center;
                box-shadow:0px 4px 10px rgba(0,0,0,0.15);
                ">

                <h2>Predicted AQI</h2>

                <h1 style="font-size:55px;">{prediction}</h1>

                <h2>{category}</h2>

                </div>
                """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("🌡 AQI Gauge Meter")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "AQI Severity Level"},
                gauge={
                    "axis": {"range": [0, 500]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 50], "color": "#4CAF50"},
                        {"range": [50, 100], "color": "#CDDC39"},
                        {"range": [100, 200], "color": "#FFC107"},
                        {"range": [200, 300], "color": "#F44336"},
                        {"range": [300, 400], "color": "#9C27B0"},
                        {"range": [400, 500], "color": "#6A1B9A"},
                    ],
                },
            )
        )

        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f"""
        <div style="
        text-align:center;
        font-size:24px;
        font-weight:bold;
        ">
        Current Status: {category}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("❤️ Health Advisory")
        for tip in advice:
            st.write("•", tip)

        st.divider()

        st.subheader("📋 Prediction Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Model Used", "XGBoost")

        with c2:
            st.metric("Input Features", "9")

        st.caption("Prediction generated using the trained XGBoost regression model.")

        st.caption(f"Prediction Time: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}")
        st.divider()

        st.subheader("🎯 Prediction Reliability")

        # Training dataset ranges
        training_ranges = {
            "PM2.5": (60, 158),
            "PM10": (125, 300),
            "NO2": (15, 44),
            "SO2": (5, 11),
            "NH3": (15, 39),
            "O3": (15, 29),
            "Temperature": (21.8, 26.2),
            "Humidity": (70, 80),
            "Wind Speed": (1.3, 2.2),
        }

        input_values = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "NH3": nh3,
            "O3": o3,
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind Speed": wind_speed,
        }

        outside_count = 0
        boundary_count = 0

        for feature, value in input_values.items():

            min_val, max_val = training_ranges[feature]

            if value < min_val or value > max_val:
                outside_count += 1

            elif (
                value <= min_val + (max_val - min_val) * 0.10
                or value >= max_val - (max_val - min_val) * 0.10
            ):
                boundary_count += 1

        # Prediction Reliability
        if outside_count == 0 and boundary_count == 0:

            st.success(
                "🟢 Prediction Reliability : HIGH\n\n"
                "All input values are within the training dataset range."
            )

        elif outside_count == 0:

            st.warning(
                "🟡 Prediction Reliability : MEDIUM\n\n"
                "Some input values are close to the training dataset boundaries."
            )

        else:

            st.error(
                "🔴 Prediction Reliability : LOW\n\n"
                "Some input values are outside the training dataset range.\n"
                "Prediction may be less reliable."
            )

        st.divider()

        st.subheader("📊 AQI Classification Scale")

        st.markdown("""
                    🟢 **Good** : 0 – 50

                    🟡 **Satisfactory** : 51 – 100

                    🟠 **Moderate** : 101 – 200

                    🔴 **Poor** : 201 – 300

                    🟣 **Very Poor** : 301 – 400

                    ⚫ **Severe** : 401+
                    """)

        st.divider()

        st.subheader("📊 Main Contributors")

        feature_names = [
            "PM2.5",
            "PM10",
            "NO₂",
            "SO₂",
            "NH₃",
            "O₃",
            "Temperature",
            "Humidity",
            "Wind Speed",
        ]

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )

        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Greens",
            text="Importance",
            template="plotly_white",
        )

        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            xaxis_title="Feature Importance",
            yaxis_title="",
            height=450,
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    # -----------------------------
    # Premium CSS
    # -----------------------------

    st.markdown(
        """
    <style>

    .hero{

        background: linear-gradient(135deg,#11998e,#38ef7d);

        padding:45px;

        border-radius:22px;

        color:white;

        text-align:center;

        box-shadow:0px 12px 35px rgba(0,0,0,0.25);

        margin-bottom:25px;

    }

    .hero h1{

        font-size:52px;

        margin-bottom:10px;

    }

    .hero h3{

        font-size:24px;

        font-weight:400;

        margin-bottom:15px;

    }

    .hero p{

        font-size:18px;

        opacity:0.95;

    }

    .divider{

        height:4px;

        background:linear-gradient(to right,#11998e,#38ef7d);

        border-radius:20px;

        margin-top:30px;

        margin-bottom:30px;

    }

    </style>
    """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # HERO SECTION
    # -----------------------------

    st.markdown(
        """

    <div class="hero">

    <h1>🌍 AQI Prediction System</h1>

    <h3>AI Powered Air Quality Intelligence Platform</h3>

    <p>
    Predict • Analyze • Visualize • Protect
    </p>

    </div>

    """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # -----------------------------
    # Introduction
    # -----------------------------

    st.markdown("## 🚀 Project Overview")

    st.write("""

This project is an intelligent Machine Learning based Air Quality Prediction System
developed using Python, Streamlit and XGBoost Regression.

It predicts the Air Quality Index (AQI) using important environmental parameters
such as PM2.5, PM10, NO₂, SO₂, NH₃, O₃, Temperature, Humidity and Wind Speed.

The dashboard is designed to provide an easy-to-understand interface for AQI
prediction, pollution analysis and interactive data visualization.

""")

    st.divider()

    # -----------------------------
    # Quick Highlights
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info("""
### 🤖 Model

XGBoost Regression

Random Forest
""")

    with col2:

        st.success("""
### 📊 Dataset

Prayagraj

Air Quality Dataset
""")

    with col3:

        st.warning("""
### ⚡ Framework

Python

Streamlit
""")

    st.divider()
    # ==========================================================
    # PROJECT STATISTICS
    # ==========================================================

    st.markdown("## 📊 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📂 Dataset",
            value="1",
            delta="Prayagraj",
        )

    with col2:
        st.metric(
            label="🤖 ML Models",
            value="2",
            delta="RF + XGBoost",
        )

    with col3:
        st.metric(
            label="📈 Features",
            value="9",
            delta="Air Parameters",
        )

    with col4:
        st.metric(
            label="⚡ Prediction",
            value="Realtime",
            delta="Instant",
        )

    st.divider()

    # ==========================================================
    # TECHNOLOGY STACK
    # ==========================================================

    st.markdown("## 🛠 Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("""
    ### 🐍 Python

    ✔ Data Processing

    ✔ Machine Learning

    ✔ Streamlit Backend
    """)

    with col2:

        st.info("""
    ### 📊 Libraries

    ✔ Pandas

    ✔ NumPy

    ✔ Plotly

    ✔ Joblib
    """)

    with col3:

        st.warning("""
    ### 🤖 AI Models

    ✔ Random Forest

    ✔ XGBoost

    ✔ Regression
    """)

    st.divider()

    # ==========================================================
    # PROJECT FEATURES
    # ==========================================================

    st.markdown("## ⭐ Key Features")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
    ✅ Real-Time AQI Prediction

    ✅ Interactive Dashboard

    ✅ AQI Gauge Meter

    ✅ Feature Importance Graph

    ✅ Health Advisory

    ✅ Reliability Checker
    """)

    with c2:

        st.markdown("""
    ✅ Pollution Visualization

    ✅ Professional UI

    ✅ Demo Prediction Mode

    ✅ Dataset Explorer

    ✅ Responsive Layout

    ✅ Interactive Charts
    """)

    st.divider()

    # ==========================================================
    # PROJECT WORKFLOW
    # ==========================================================

    st.markdown("## 🔄 Project Workflow")
    st.info("""
    📂 Dataset Collection

    ⬇

    🧹 Data Cleaning

    ⬇

    📊 Data Visualization

    ⬇

    🤖 Model Training

    ⬇

    📈 AQI Prediction

    ⬇

    🌍 Dashboard Deployment
    """)
    # ==========================================================
    # DEVELOPER PROFILE SECTION
    # ==========================================================

    st.divider()

    st.subheader("👩‍💻 Developer Profile")

    # Load Profile Image

    profile_image = Image.open("Images/Anjalipic.jpg")

    # Main Card

    st.markdown(
        """
    <div style="
    background:linear-gradient(135deg,#f1f8e9,#ffffff);
    padding:30px;
    border-radius:25px;
    border:1px solid #dcedc8;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
    ">
    """,
        unsafe_allow_html=True,
    )

    # Photo + Details Layout

    col1, col2 = st.columns([1, 3])

    with col1:

        st.image(profile_image, width=170)

    with col2:

        st.markdown(
            """
        <h1 style="
        color:#2e7d32;
        margin-bottom:5px;
        ">
        Anjali Tiwari
        </h1>


        <p style="
        font-size:17px;
        color:#555;
        ">
        CSE (Data Science) Student |
        Python Developer |
        Machine Learning Enthusiast
        </p>


        <hr style="
        border:0.5px solid #dcedc8;
        ">


        <p style="
        font-size:16px;
        color:#444;
        line-height:1.6;
        ">

        Passionate about building intelligent applications
        using Python, Data Science and Machine Learning.

        This AQI Prediction System transforms real-world
        environmental data into meaningful insights through
        machine learning models and interactive visualization.

        </p>



        <h3 style="color:#388e3c;">
        🚀 Skills & Expertise
        </h3>


        <p style="
        font-size:16px;
        color:#555;
        ">

        🐍 Python Programming <br>
        📊 Pandas & NumPy <br>
        🤖 Machine Learning (XGBoost, Random Forest) <br>
        📈 Data Visualization (Plotly) <br>
        🌐 Streamlit Dashboard Development <br>
        🗄 SQL & Data Handling

        </p>



        <h3 style="color:#388e3c;">
        📌 Project Contribution
        </h3>


        <p style="
        font-size:16px;
        color:#555;
        line-height:1.5;
        ">

        Designed and developed the AQI Prediction Dashboard
        including data preprocessing, machine learning model
        integration and interactive visualization.

        </p>



        <h3 style="color:#388e3c;">
        📅 Development Journey
        </h3>


        <p style="
        font-size:16px;
        color:#555;
        ">

        📂 Dataset Collection →
        🧹 Data Cleaning →
        🤖 Model Training →
        📊 Dashboard Development →
        🚀 Deployment

        </p>



        <h3 style="color:#388e3c;">
        🌱 Learning Outcome
        </h3>


        <p style="
        font-size:16px;
        color:#555;
        ">

        Practical experience in Machine Learning workflow,
        real-world datasets, model implementation and
        interactive Data Science applications.

        </p>



        <h3 style="color:#388e3c;">
        🔗 Connect With Me
        </h3>


        <a href="https://github.com/Anjalitiwari99"
        target="_blank"
        style="
        text-decoration:none;
        color:#24292e;
        font-size:18px;
        font-weight:600;
        ">

        <i class="fa-brands fa-github"></i>
        GitHub

        </a>


        &nbsp;&nbsp;&nbsp;



        <a href="https://www.linkedin.com/in/anjali-tiwari-91aa37318"
        target="_blank"
        style="
        text-decoration:none;
        color:#0a66c2;
        font-size:18px;
        font-weight:600;
        ">

        <i class="fa-brands fa-linkedin"></i>
        LinkedIn

        </a>


        &nbsp;&nbsp;&nbsp;



        <a href="https://x.com/anjali_tiwari95"
        target="_blank"
        style="
        text-decoration:none;
        color:#000000;
        font-size:18px;
        font-weight:600;
        ">

        <i class="fa-brands fa-x-twitter"></i>
        X

        </a>


        """,
            unsafe_allow_html=True,
        )

    # Close Card

    st.markdown(
        """
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ==========================================================
    # CHAPTER 5 : FINAL MINIMAL FOOTER
    # ==========================================================

    st.divider()

    st.markdown(
        """
    <style>

    .footer-box {

    background:linear-gradient(90deg,#f1f8e9,#ffffff);

    padding:15px;

    border-radius:15px;

    border:1px solid #dcedc8;

    text-align:center;

    font-size:15px;

    color:#666;

    box-shadow:0px 4px 15px rgba(0,0,0,0.05);

    }


    .footer-box strong {

    color:#2e7d32;

    }


    </style>



    <div class="footer-box">


    🌱 <strong>AQI Prediction System</strong>

    &nbsp; | &nbsp;

    Developed by <strong>Anjali Tiwari</strong>

    &nbsp; | &nbsp;

    © 2026


    </div>


    """,
        unsafe_allow_html=True,
    )
