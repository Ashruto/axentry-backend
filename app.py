from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import threading
import sqlite3
import requests

# -------------------------------
# FLASK SERVER
# -------------------------------

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": ["https://v0-axentry-saas.vercel.app"]}},
    supports_credentials=True,
)

# -------------------------------
# DATABASE SETUP
# -------------------------------

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

# -------------------------------
# AI MODEL + CAMERA (LOCAL ONLY)
# -------------------------------

if os.environ.get("RENDER") != "true":
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    BUFFER_SECONDS = 10
    buffer = deque(maxlen=FPS * BUFFER_SECONDS)
else:
    FPS = 30
    BUFFER_SECONDS = 10
    buffer = None

# -------------------------------
# STATE VARIABLES
# -------------------------------

scanning = False
scan_start = None
flagged = False
primary_id = None
secondary_detect_time = None
scan_triggered = False
TOLERANCE_SECONDS = 1.0

last_status = None
status_color = (0, 255, 0)
status_display_until = 0

os.makedirs("clips", exist_ok=True)

# -------------------------------
# ROUTES
# -------------------------------

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


@app.route("/scan", methods=["POST"])
def trigger_scan():
    global scan_triggered
    scan_triggered = True
    return {"status": "scan received"}, 200


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


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory("clips", filename)


@app.route("/dashboard")
def dashboard():
    return "Dashboard running"


# -------------------------------
# RUN SERVER (RENDER SIDE)
# -------------------------------

if os.environ.get("RENDER") == "true":
    print("🌍 Running Render API server...")
else:
    def run_server():
        app.run(host="0.0.0.0", port=5055)

    threading.Thread(target=run_server, daemon=True).start()

    print("System Ready | Awaiting biometric scan on /scan")
    print("Dashboard: http://localhost:5055/dashboard")

# -------------------------------
# MAIN LOOP (LOCAL ONLY)
# -------------------------------

if os.environ.get("RENDER") != "true":

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

        cv2.imshow("Axentry Access Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()
