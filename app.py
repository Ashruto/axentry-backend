from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os
from flask import Flask, jsonify, send_from_directory
import threading
import sqlite3
import requests  # 🔥 ADDED
from flask_cors import CORS


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
if os.environ.get("RENDER") != "true":
 model = YOLO("yolov8n.pt")
 cap = cv2.VideoCapture(0)

FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
BUFFER_SECONDS = 10
buffer = deque(maxlen=FPS * BUFFER_SECONDS)

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
# FLASK SERVER
# -------------------------------
app = Flask(__name__)
CORS(app)


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

 h, w, _ = frame.shape

 dx1 = int(w * 0.25)
 dx2 = int(w * 0.75)
 dy1 = 0
 dy2 = h

 buffer.append(frame.copy())
 cv2.rectangle(frame,(dx1,dy1),(dx2,dy2),(255,0,0),2)

 results = model.track(frame, conf=0.5, classes=[0], persist=True, verbose=False)

 ids=[]
 boxes=[]
 if results[0].boxes.id is not None:
  ids = results[0].boxes.id.cpu().tolist()
  boxes = results[0].boxes.xyxy.cpu().tolist()

 valid_ids=set()

 for box,tid in zip(boxes,ids):
  x1,y1,x2,y2=map(int,box)
  cx=(x1+x2)//2
  cy=(y1+y2)//2

  cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
  cv2.putText(frame,f"ID {int(tid)}",(x1,y1-8),
   cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

  if dx1<cx<dx2 and dy1<cy<dy2:
   valid_ids.add(tid)

 if scan_triggered and not scanning:
  scanning=True
  scan_start=time.time()
  flagged=False
  primary_id=None
  secondary_detect_time=None
  scan_triggered=False

 if scanning:
  elapsed=time.time()-scan_start

  if primary_id is None and len(valid_ids)>=1:
   primary_id=list(valid_ids)[0]

  if primary_id is not None:
   secondary_ids=valid_ids-{primary_id}

   if len(secondary_ids)>0:
    if secondary_detect_time is None:
     secondary_detect_time=time.time()
    elif time.time()-secondary_detect_time>=TOLERANCE_SECONDS:
     flagged=True
   else:
    secondary_detect_time=None

  cv2.putText(frame,
   f"Scanning... {round(3-elapsed,1)}s",
   (20,40),
   cv2.FONT_HERSHEY_SIMPLEX,
   1,(0,255,255),2)

  if elapsed>3:
   scanning=False
   readable_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   if flagged:
    last_status="UNAUTHORIZED ENTRY DETECTED"
    status_color=(0,0,255)

    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    filename=f"event_{ts}.mp4"
    path=os.path.join("clips",filename)

    h,w,_=frame.shape
    out=cv2.VideoWriter(path,
     cv2.VideoWriter_fourcc(*"mp4v"),
     FPS,(w,h))

    for f in buffer:
     out.write(f)
    out.release()

    # 🔥 CHANGED HERE
    requests.post(
     "https://axentry-backend.onrender.com/insert",
     json={
      "timestamp": readable_time,
      "status": "UNAUTHORIZED",
      "clip_path": f"clips/{filename}",
      "camera_id": "CAM_01"
     }
    )

   else:
    last_status="ACCESS VERIFIED"
    status_color=(0,200,0)

    # 🔥 CHANGED HERE
    requests.post(
     "https://axentry-backend.onrender.com/insert",
     json={
      "timestamp": readable_time,
      "status": "VERIFIED",
      "clip_path": None,
      "camera_id": "CAM_01"
     }
    )

   status_display_until=time.time()+3

 if time.time()<status_display_until and last_status:
  cv2.putText(frame,last_status,(20,80),
   cv2.FONT_HERSHEY_SIMPLEX,
   1.1,status_color,3)

 cv2.imshow("Axentry Access Verification",frame)

 if cv2.waitKey(1) & 0xFF==ord("q"):
  break

cap.release()
cv2.destroyAllWindows()
