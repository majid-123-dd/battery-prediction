import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Mobile Battery Life Predictor",
    page_icon="🔋",
    layout="wide"
)

# -------------------------
# SET BACKGROUND
# -------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/background.jpg")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# TITLE
# -------------------------
st.title("🔋 Mobile Battery Life Predictor")
st.image("images/battery_banner.png", use_container_width=True)
st.write("Predict daily battery life based on usage behavior.")

# -------------------------
# INPUT SECTION
# -------------------------
col1, col2 = st.columns(2)

with col1:
    screen_time = st.slider("📱 Screen Time (hours)", 1, 15, 6)
    gaming_hours = st.slider("🎮 Gaming Hours", 0, 10, 2)
    brightness = st.slider("💡 Brightness Level (%)", 10, 100, 70)

with col2:
    battery_capacity = st.number_input("🔋 Battery Capacity (mAh)", 2000, 6000, 4000)
    background_apps = st.slider("📊 Background Apps Running", 0, 50, 10)
    app_usage = st.slider("📦 Number of Apps Used Daily", 1, 40, 12)

# -------------------------
# PREDICTION
# -------------------------
if st.button("🚀 Predict Battery Life"):

    input_data = np.array([[screen_time, gaming_hours, brightness,
                            battery_capacity, background_apps, app_usage]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("🔍 Prediction Result")

    st.success(f"Estimated Battery Life: {prediction:.2f} hours")

    # Battery Level Indicator
    if prediction > 14:
        st.image("images/high.png", width=150)
        st.success("🟢 Excellent Battery Performance")

    elif prediction > 8:
        st.image("images/medium.png", width=150)
        st.warning("🟡 Moderate Battery Drain")

    else:
        st.image("images/low.png", width=150)
        st.error("🔴 High Battery Drain")

    # -------------------------
    # SMART RECOMMENDATIONS
    # -------------------------
    st.subheader("💡 Optimization Tips")

    if gaming_hours > 3:
        st.warning("Reduce gaming time to improve battery life.")

    if brightness > 80:
        st.warning("Lower brightness level to save power.")

    if background_apps > 20:
        st.warning("Close unused background apps.")

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    st.subheader("📊 Feature Importance")

    features = ["Screen Time", "Gaming Hours", "Brightness",
                "Battery Capacity", "Background Apps", "App Usage"]

    try:
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(features, importance)
        st.pyplot(fig)
    except:
        st.info("Feature importance available only for Decision Tree model.")
