from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os
from flask import Flask, jsonify, send_from_directory
import threading
import sqlite3

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
# AI MODEL + CAMERA
# -------------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
BUFFER_SECONDS = 10
buffer = deque(maxlen=FPS * BUFFER_SECONDS)

DOOR_REGION = (200, 0, 450, 480)

# -------------------------------
# STATE VARIABLES
# -------------------------------
scanning = False
scan_start = None
flagged = False
primary_id = None
secondary_detect_time = None
TOLERANCE_SECONDS = 1.0

os.makedirs("clips", exist_ok=True)

# -------------------------------
# FLASK SERVER
# -------------------------------
app = Flask(__name__)
scan_triggered = False


@app.route("/scan", methods=["POST"])
def trigger_scan():
    global scan_triggered
    scan_triggered = True
    return {"status": "scan received"}, 200


@app.route("/events", methods=["GET"])
def get_events():
    cursor.execute("SELECT * FROM events ORDER BY id DESC")
    rows = cursor.fetchall()

    events = []
    for row in rows:
        events.append({
            "id": row[0],
            "timestamp": row[1],
            "status": row[2],
            "clip_path": row[3],
            "camera_id": row[4]
        })

    return jsonify(events)


# ✅ FIXED CLIP ROUTE
@app.route("/clips/<path:filename>")
def serve_clip(filename):
    full_path = os.path.join("clips", filename)

    if not os.path.exists(full_path):
        return {"error": "File not found"}, 404

    return send_from_directory("clips", filename)


# -------------------------------
# DASHBOARD
# -------------------------------
@app.route("/dashboard")
def dashboard():
    return """
    <html>
    <head>
    <title>Axentry Dashboard</title>
    <style>
    body {font-family: Arial; background:#0f172a; color:white; padding:20px;}
    .cards {display:flex; gap:20px; margin-bottom:20px;}
    .card {background:#1e293b; padding:20px; border-radius:10px; flex:1;}
    table {width:100%; border-collapse:collapse; background:#1e293b;}
    th,td {padding:12px;}
    th {background:#334155;}
    .verified {color:#22c55e; font-weight:bold;}
    .unauthorized {color:#ef4444; font-weight:bold;}
    a {color:#38bdf8;}
    </style>
    </head>
    <body>
    <h1>Axentry Access Dashboard</h1>

    <div class="cards">
        <div class="card"><h3>Total</h3><p id="total">0</p></div>
        <div class="card"><h3>Verified</h3><p id="verified">0</p></div>
        <div class="card"><h3>Unauthorized</h3><p id="unauthorized">0</p></div>
        <div class="card"><h3>Violation %</h3><p id="rate">0%</p></div>
    </div>

    <table>
    <thead>
    <tr>
    <th>ID</th><th>Timestamp</th><th>Status</th><th>Camera</th><th>Clip</th>
    </tr>
    </thead>
    <tbody id="table"></tbody>
    </table>

    <script>
    async function load(){
        const res = await fetch('/events');
        const data = await res.json();

        let verified = 0;
        let unauthorized = 0;

        const table = document.getElementById("table");
        table.innerHTML = "";

        data.forEach(e => {
            if(e.status==="UNAUTHORIZED") unauthorized++;
            else verified++;

            const row = document.createElement("tr");

            row.innerHTML = `
            <td>${e.id}</td>
            <td>${e.timestamp}</td>
            <td class="${e.status==="UNAUTHORIZED"?"unauthorized":"verified"}">
                ${e.status}
            </td>
            <td>${e.camera_id}</td>
            <td>
                ${
                    e.clip_path
                    ? `<a href="/clips/${e.clip_path.replace("clips/","")}" target="_blank">View Clip</a>`
                    : "-"
                }
            </td>
            `;
            table.appendChild(row);
        });

        document.getElementById("total").innerText = data.length;
        document.getElementById("verified").innerText = verified;
        document.getElementById("unauthorized").innerText = unauthorized;

        const rate = data.length ? ((unauthorized/data.length)*100).toFixed(1) : 0;
        document.getElementById("rate").innerText = rate + "%";
    }

    load();
    setInterval(load, 5000);
    </script>
    </body>
    </html>
    """


def run_server():
    app.run(host="0.0.0.0", port=5055)


threading.Thread(target=run_server, daemon=True).start()

print("System Ready | Awaiting biometric scan on /scan")
print("Dashboard: http://localhost:5055/dashboard")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    buffer.append(frame.copy())
    dx1, dy1, dx2, dy2 = DOOR_REGION

    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255,0,0), 2)

    results = model.track(frame, conf=0.5, classes=[0], persist=True, verbose=False)

    ids = []
    boxes = []

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

    valid_ids = set()

    for box, track_id in zip(boxes, ids):
        x1,y1,x2,y2 = map(int, box)
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        if dx1 < cx < dx2 and dy1 < cy < dy2:
            valid_ids.add(track_id)

    if scan_triggered and not scanning:
        scanning = True
        scan_start = time.time()
        flagged = False
        primary_id = None
        secondary_detect_time = None
        scan_triggered = False

    if scanning:
        elapsed = time.time() - scan_start

        if primary_id is None and len(valid_ids) >= 1:
            primary_id = list(valid_ids)[0]

        if primary_id is not None:
            secondary_ids = valid_ids - {primary_id}

            if len(secondary_ids) > 0:
                if secondary_detect_time is None:
                    secondary_detect_time = time.time()
                elif time.time() - secondary_detect_time >= TOLERANCE_SECONDS:
                    flagged = True
            else:
                secondary_detect_time = None

        if elapsed > 3:
            scanning = False
            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if flagged:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"event_{ts}.mp4"
                path = os.path.join("clips", filename)

                h,w,_ = frame.shape
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w,h))
                for f in buffer:
                    out.write(f)
                out.release()
                time.sleep(0.2)

                cursor.execute("""
                INSERT INTO events (timestamp, status, clip_path, camera_id)
                VALUES (?, ?, ?, ?)
                """, (readable_time, "UNAUTHORIZED", f"clips/{filename}", "CAM_01"))
            else:
                cursor.execute("""
                INSERT INTO events (timestamp, status, clip_path, camera_id)
                VALUES (?, ?, ?, ?)
                """, (readable_time, "VERIFIED", None, "CAM_01"))

            conn.commit()

    cv2.imshow("Axentry Access Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
