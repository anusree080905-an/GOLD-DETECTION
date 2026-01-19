import cv2
import numpy as np
import os
import time
from datetime import datetime

# CAMERA CONFIG 
IP_CAMERA_URL = "http://192.168.1.34:8080/videofeed"
cap = cv2.VideoCapture(IP_CAMERA_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#  OUTPUT FOLDER 
OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RECORDING CONTROL 
video_writer = None
recording = False
last_gold_time = None
STOP_DELAY = 2
frame_number = 0

#  CUSTOMER MOTION 
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

CUSTOMER_Y_START = 0
CUSTOMER_Y_END = height // 2
last_customer_motion_time = None
CUSTOMER_HOLD_TIME = 3

#  GOLD COUNTER (CENTERED) 
COUNTER_W = int(width * 0.6)
COUNTER_H = int(height * 0.35)

COUNTER_X1 = (width - COUNTER_W) // 2
COUNTER_Y1 = (height - COUNTER_H) // 2
COUNTER_X2 = COUNTER_X1 + COUNTER_W
COUNTER_Y2 = COUNTER_Y1 + COUNTER_H

print("✅ Bank Gold Loan Monitoring System Running")

#  MAIN LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    now = time.time()

    #  GOLD COUNTER ROI 
    counter_roi = frame[COUNTER_Y1:COUNTER_Y2, COUNTER_X1:COUNTER_X2]

    hsv = cv2.cvtColor(counter_roi, cv2.COLOR_BGR2HSV)

    LOWER_GOLD = np.array([12, 85, 70])
    UPPER_GOLD = np.array([38, 255, 255])

    gold_mask = cv2.inRange(hsv, LOWER_GOLD, UPPER_GOLD)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel)
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)

    gold_pixels = cv2.countNonZero(gold_mask)

    gray = cv2.cvtColor(counter_roi, cv2.COLOR_BGR2GRAY)
    shine_mask = cv2.inRange(gray, 180, 225)
    shine_pixels = cv2.countNonZero(shine_mask)

    contours, _ = cv2.findContours(
        gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_gold = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 350:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 3.5:
                valid_gold = True
                break

    gold_detected = (
        valid_gold and
        gold_pixels > 1000 and
        shine_pixels > 60
    )

    #  CUSTOMER MOVEMENT 
    customer_region = frame[CUSTOMER_Y_START:CUSTOMER_Y_END, :]
    fg_mask = bg_subtractor.apply(customer_region)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    motion_pixels = cv2.countNonZero(fg_mask)

    if motion_pixels > 1200:
        customer_present = True
        last_customer_motion_time = now
    else:
        customer_present = (
            last_customer_motion_time and
            now - last_customer_motion_time < CUSTOMER_HOLD_TIME
        )

    # RECORDING 
    if gold_detected and not recording:
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        recording = True
        print(f"🎥 RECORDING STARTED: {filename}")

    if gold_detected:
        last_gold_time = now

    if recording and last_gold_time and now - last_gold_time > STOP_DELAY:
        video_writer.release()
        recording = False
        last_gold_time = None
        print("⏹ RECORDING STOPPED")

    #  OVERLAY 
    timestamp = datetime.now().strftime("%H:%M:%S")

    cv2.putText(frame, f"FRAME: {frame_number}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"TIME: {timestamp}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status_text = "GOLD DETECTED - RECORDING" if recording else "NO GOLD - NOT RECORDING"
    status_color = (0, 0, 255) if recording else (0, 255, 0)

    cv2.putText(frame, status_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if customer_present:
        cv2.putText(frame, "CUSTOMER PRESENT", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    #  GOLD COUNTER BOX
    cv2.rectangle(frame,
                  (COUNTER_X1, COUNTER_Y1),
                  (COUNTER_X2, COUNTER_Y2),
                  (0, 255, 255), 2)

    cv2.putText(frame, "GOLD COUNTER AREA",
                (COUNTER_X1, COUNTER_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #  SAVE 
    if recording:
        video_writer.write(frame)

    cv2.imshow("Bank Gold Loan Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  CLEANUP
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()