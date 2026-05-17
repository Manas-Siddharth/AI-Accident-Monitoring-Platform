import streamlit as st
import cv2
import pandas as pd
import os
from ultralytics import YOLO
from PIL import Image
import plotly.express as px


# Load YOLO model
model = YOLO("yolov8n.pt")

# Page setup
st.set_page_config(
    page_title="AI Accident Monitoring Platform",
    layout="wide"
)

st.markdown("""
<h1 style='text-align: center;
color: #00FFAA;
font-size: 50px;
font-weight: bold;'>

🚨 AI ACCIDENT MONITORING PLATFORM 🚨

</h1>
""",
unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align: center;
color: white;'>

Real-Time Intelligent Traffic Surveillance & Emergency Response System

</h4>
""",
unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.title("System Controls")

run = st.sidebar.checkbox("Start Monitoring")

st.sidebar.markdown("---")


# Analytics storage
# Store analytics history
if "vehicle_history" not in st.session_state:
    st.session_state.vehicle_history = []

if "speed_history" not in st.session_state:
    st.session_state.speed_history = []
# Dashboard metrics
metric1, metric2, metric3, metric4 = st.columns(4)

vehicle_count_placeholder = metric1.empty()
accident_count_placeholder = metric2.empty()
speed_placeholder = metric3.empty()
alert_placeholder = metric4.empty()

st.markdown("---")

# Video section
st.subheader("📹 Live Vehicle Detection Feed")

frame_placeholder = st.empty()

# Vehicle classes
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
total_accidents = 0
# Start monitoring
if run:

    cap = cv2.VideoCapture("dataset/traffic.mp4")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.warning("Video Ended")
            break

        # Resize frame
        frame = cv2.resize(frame, (1280, 720))

        # YOLO detection
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        boxes = results[0].boxes
        vehicle_count = 0
        total_speed = 0

        if boxes.id is not None:

            track_ids = boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):

                cls_id = int(box.cls[0])

                class_name = model.names[cls_id]

                if class_name in vehicle_classes:
                    vehicle_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    confidence = float(box.conf[0])

                    # Draw rectangle
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label
                    speed = int(confidence * 100)

                    total_speed += speed

                    label = f"ID {track_id} | {class_name} | Speed: {speed}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        # Average speed
        avg_speed = 0
        # Store analytics data
        # Store live analytics
        st.session_state.vehicle_history.append(vehicle_count)
        st.session_state.speed_history.append(avg_speed)

        # Keep last 30 values
        st.session_state.vehicle_history = \
            st.session_state.vehicle_history[-30:]

        st.session_state.speed_history = \
            st.session_state.speed_history[-30:]

        if vehicle_count > 0:
            avg_speed = total_speed // vehicle_count

        # Update dashboard metrics
        vehicle_count_placeholder.metric(
            "Vehicles Detected",
            vehicle_count
        )

        accident_count_placeholder.metric(
            "Accidents Logged",
            len(pd.read_csv("outputs/accident_log.csv"))
        )

        speed_placeholder.metric(
            "Average Speed",
            avg_speed
        )

        # AI Risk Analysis
        risk_level = "LOW"

        if vehicle_count > 15 and avg_speed > 60:
            risk_level = "HIGH"

        elif vehicle_count > 10 and avg_speed > 40:
            risk_level = "MEDIUM"

        elif vehicle_count > 20 and avg_speed > 80:
            risk_level = "CRITICAL"

        # Update alert panel
        alert_placeholder.metric(
            "AI Risk Level",
            risk_level
        )
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame
        frame_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

st.markdown("---")

st.subheader("📈 Live AI Traffic Analytics")

graph_col1, graph_col2 = st.columns(2)

# Traffic Density Graph
traffic_df = pd.DataFrame({
    "Frame": list(range(len(st.session_state.vehicle_history))),
    "Vehicles": st.session_state.vehicle_history
})

fig1 = px.line(
    traffic_df,
    x="Frame",
    y="Vehicles",
    title="Traffic Density Analysis"
)

graph_col1.plotly_chart(
    fig1,
    use_container_width=True
)

# Speed Analysis Graph
speed_df = pd.DataFrame({
    "Frame": list(range(len(st.session_state.speed_history))),
    "Speed": st.session_state.speed_history
})

fig2 = px.line(
    speed_df,
    x="Frame",
    y="Speed",
    title="Average Speed Analysis"
)

graph_col2.plotly_chart(
    fig2,
    use_container_width=True
)

# Accident history
st.subheader("📄 Accident History")

log_file = "outputs/accident_log.csv"

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    st.dataframe(df,
                 use_container_width=True)

else:

    st.warning("No accident history found.")

st.markdown("---")

# Latest accident image
st.subheader("📸 Latest Accident Screenshot")

if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    if len(df) > 0:

        latest_image = df.iloc[-1]["Screenshot"]

        if os.path.exists(latest_image):

            image = Image.open(latest_image)

            st.image(
                image,
                caption="Latest Accident Detection",
                use_container_width=True
            )

st.success("AI Surveillance System Running Successfully ✅")
