from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os
from flask import Flask, jsonify, send_from_directory, request
import threading
import sqlite3
import requests   # 🔥 NEW

# ===============================
# DATABASE (ONLY USED ON RENDER)
# ===============================

app = Flask(__name__)

conn = sqlite3.connect("events.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    status TEXT,
    clip_path TEXT,
    camera_id TEXT
)
""")
conn.commit()

# ===============================
# RENDER INSERT ROUTE (CLOUD)
# ===============================

@app.route("/insert", methods=["POST"])
def insert_event():
    data = request.json

    cursor.execute("""
        INSERT INTO events (timestamp, status, clip_path, camera_id)
        VALUES (?, ?, ?, ?)
    """, (
        data["timestamp"],
        data["status"],
        data.get("clip_path"),
        data["camera_id"]
    ))

    conn.commit()
    return {"success": True}

# ===============================
# EVENTS FETCH ROUTE
# ===============================

@app.route("/events")
def get_events():
    cursor.execute("SELECT * FROM events ORDER BY id DESC")
    rows = cursor.fetchall()

    return jsonify([
        {
            "id": r[0],
            "timestamp": r[1],
            "status": r[2],
            "clip_path": r[3],
            "camera_id": r[4]
        } for r in rows
    ])

# ===============================
# AI + CAMERA (ONLY RUN LOCALLY)
# ===============================

if os.environ.get("RENDER") != "true":

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    BUFFER_SECONDS = 10
    buffer = deque(maxlen=FPS * BUFFER_SECONDS)

    print("🚀 Local AI running. Sending events to cloud...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame.copy())

        results = model.track(frame, conf=0.5, classes=[0], persist=True, verbose=False)

        unauthorized_detected = False

        if results[0].boxes.id is not None:
            if len(results[0].boxes.id.cpu().tolist()) > 1:
                unauthorized_detected = True

        readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if unauthorized_detected:
            print("❌ UNAUTHORIZED DETECTED")

            requests.post(
                "https://axentry-backend.onrender.com/insert",
                json={
                    "timestamp": readable_time,
                    "status": "UNAUTHORIZED",
                    "clip_path": None,
                    "camera_id": "CAM_01"
                }
            )
        else:
            print("✅ VERIFIED")

            requests.post(
                "https://axentry-backend.onrender.com/insert",
                json={
                    "timestamp": readable_time,
                    "status": "VERIFIED",
                    "clip_path": None,
                    "camera_id": "CAM_01"
                }
            )

        time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()

# ===============================
# RUN FLASK (RENDER SIDE)
# ===============================

if os.environ.get("RENDER") == "true":
    print("🌍 Running Render API server...")
