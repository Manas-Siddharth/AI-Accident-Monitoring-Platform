import streamlit as st
import pandas as pd
import os
from PIL import Image

# Dashboard title
st.set_page_config(page_title="AI Accident Detection Dashboard",
                   layout="wide")

st.title("🚨 AI-Powered Accident Detection Dashboard")

st.markdown("---")

# System status
st.subheader("🟢 System Status")

col1, col2, col3 = st.columns(3)

col1.metric("System", "ACTIVE")
col2.metric("Camera", "ONLINE")
col3.metric("Alert System", "RUNNING")

st.markdown("---")

# Load accident log
log_file = "outputs/accident_log.csv"

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    # Statistics
    st.subheader("📊 Live Statistics")

    total_accidents = len(df)

    col1, col2 = st.columns(2)

    col1.metric("Total Accidents Detected",
                total_accidents)

    col2.metric("Monitoring Status",
                "ACTIVE")

    st.markdown("---")

    # Accident history table
    st.subheader("📄 Accident History")

    st.dataframe(df,
                 use_container_width=True)

    st.markdown("---")

    # Latest accident image
    st.subheader("📸 Latest Accident Screenshot")

    if total_accidents > 0:

        latest_image_path = df.iloc[-1]["Screenshot"]

        if os.path.exists(latest_image_path):

            image = Image.open(latest_image_path)

            st.image(image,
                     caption="Latest Accident Detection",
                     use_container_width=True)

        else:
            st.warning("Screenshot file not found.")

else:

    st.error("No accident log found.")

st.markdown("---")

st.success("AI Surveillance System Running Successfully ✅")