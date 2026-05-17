# 🚨 AI-Powered Accident Monitoring Platform

An intelligent real-time traffic surveillance and accident detection platform built using Artificial Intelligence, Computer Vision, and Data Analytics.

This system uses YOLOv8 object detection, vehicle tracking, speed estimation, accident monitoring logic, and a futuristic Streamlit dashboard to monitor traffic activity in real time.

---

# 📌 Project Overview

The AI-Powered Accident Monitoring Platform is designed to improve road safety through intelligent traffic surveillance. The system continuously monitors vehicle movement using computer vision techniques and detects potential accidents based on vehicle behavior and motion analysis.

The platform provides:
- Real-time vehicle detection
- Vehicle tracking with IDs
- Speed estimation
- AI-based risk analysis
- Accident logging
- Emergency alert monitoring
- Live dashboard analytics
- Traffic visualization graphs

The project demonstrates how Artificial Intelligence can be integrated into smart city infrastructure for traffic safety and emergency response systems.

---

# 🚀 Features

## ✅ Real-Time Vehicle Detection
- Detects cars, buses, trucks, and motorcycles
- Uses YOLOv8 object detection model

## ✅ Vehicle Tracking
- Assigns unique tracking IDs to vehicles
- Tracks movement across frames

## ✅ Speed Estimation
- Estimates vehicle speed dynamically
- Displays live speed values

## ✅ AI Risk Prediction
- Predicts traffic risk levels:
  - LOW
  - MEDIUM
  - HIGH
  - CRITICAL

## ✅ Accident Monitoring
- Detects possible accident scenarios
- Captures screenshots automatically

## ✅ Emergency Alert System
- Triggers alerts when abnormal activity is detected

## ✅ Live Analytics Dashboard
- Real-time monitoring interface using Streamlit
- Dynamic traffic analytics graphs

## ✅ Data Logging
- Stores accident details in CSV format
- Maintains accident history records

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| YOLOv8 | Object Detection |
| OpenCV | Computer Vision |
| Streamlit | Dashboard UI |
| Plotly | Analytics Graphs |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Pillow | Image Processing |

---

# 🏗️ System Architecture

```text
Traffic Video Feed
        ↓
YOLOv8 Vehicle Detection
        ↓
Vehicle Tracking
        ↓
Speed Estimation
        ↓
Risk Analysis Engine
        ↓
Accident Detection Logic
        ↓
Dashboard Visualization
        ↓
CSV Logging & Alerts
