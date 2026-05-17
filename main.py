from ultralytics import YOLO
import cv2
import os
import time
import pygame
import csv
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize pygame mixer
pygame.mixer.init()

# Load alarm sound
alarm_sound = pygame.mixer.Sound("sounds/alarm.wav")


# Open video
cap = cv2.VideoCapture("dataset/road cars.mp4") 

# Vehicle classes
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Create window
cv2.namedWindow("Accident Detection System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Accident Detection System", 1280, 720)

# Create output folder
os.makedirs("outputs/accidents", exist_ok=True)
# CSV log file
log_file = "outputs/accident_log.csv"

# Create CSV file with headers
if not os.path.exists(log_file):

    with open(log_file, mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            "Accident ID",
            "Date",
            "Time",
            "Screenshot"
        ])

# Function to check overlap
def boxes_overlap(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    # Check overlap
    return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)

# Prevent repeated screenshots
last_accident_time = 0
# Store previous vehicle positions
vehicle_positions = {}
vehicle_speeds = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    boxes_data = []

    boxes = results[0].boxes

    if boxes.id is not None:

        track_ids = boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                speed = 0

                # Check previous position
                if track_id in vehicle_positions:

                    prev_x, prev_y = vehicle_positions[track_id]

                    # Simple speed calculation
                    speed = int(((center_x - prev_x) ** 2 +
                                (center_y - prev_y) ** 2) ** 0.5)

                # Update vehicle position
                vehicle_positions[track_id] = (center_x, center_y)
                vehicle_speeds[track_id] = speed

                boxes_data.append((x1, y1, x2, y2))

                # Draw vehicle box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

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

    # Accident detection logic
    accident_detected = False

    for i in range(len(boxes_data)):
        for j in range(i + 1, len(boxes_data)):

            box1 = boxes_data[i]
            box2 = boxes_data[j]

            if boxes_overlap(box1, box2):

                # Get speeds
                speed_values = list(vehicle_speeds.values())

                if len(speed_values) >= 2:

                    speed1 = speed_values[i]
                    speed2 = speed_values[j]

                    # Detect sudden stop/collision
                    if speed1 < 5 and speed2 < 5:

                        accident_detected = True

    # If accident detected
    if accident_detected:

        # Play alarm
        if not pygame.mixer.get_busy():
            alarm_sound.play()

        # Alert text
        cv2.putText(
            frame,
            "ACCIDENT DETECTED!",
            (400, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

        # Save screenshot every 5 seconds
        current_time = time.time()

        if current_time - last_accident_time > 5:

            filename = f"outputs/accidents/accident_{int(current_time)}.jpg"

            cv2.imwrite(filename, frame)
            # Current date and time
            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time_now = now.strftime("%H:%M:%S")

            # Accident ID
            accident_id = int(current_time)

            # Save to CSV
            with open(log_file, mode='a', newline='') as file:

                writer = csv.writer(file)

                writer.writerow([
                    accident_id,
                    date,
                    time_now,
                    filename
                ])

            print(f"Accident screenshot saved: {filename}")

            last_accident_time = current_time

    # Show frame
    cv2.imshow("Accident Detection System", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()