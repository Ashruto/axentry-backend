from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os
from flask import Flask
import threading

# -------------------------------
# AI MODEL + CAMERA
# -------------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
BUFFER_SECONDS = 10
buffer = deque(maxlen=FPS * BUFFER_SECONDS)

# -------------------------------
# ENTRY ZONE (ADJUST FOR CAMERA)
# -------------------------------
DOOR_REGION = (200, 0, 450, 480)

# -------------------------------
# SCAN STATE VARIABLES
# -------------------------------
scanning = False
scan_start = None
flagged = False
primary_id = None
secondary_detect_time = None
TOLERANCE_SECONDS = 1.0

# -------------------------------
# CLIP STORAGE
# -------------------------------
os.makedirs("clips", exist_ok=True)

# -------------------------------
# BIOMETRIC API SERVER
# -------------------------------
app = Flask(__name__)
scan_triggered = False

@app.route("/scan", methods=["POST"])
def trigger_scan():
    global scan_triggered
    scan_triggered = True
    return {"status": "scan received"}, 200

def run_server():
    app.run(host="0.0.0.0", port=5055)

threading.Thread(target=run_server, daemon=True).start()

print("System Ready | Awaiting biometric scan on /scan")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    buffer.append(frame.copy())

    dx1, dy1, dx2, dy2 = DOOR_REGION

    # Draw entry zone
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
    cv2.putText(frame, "ENTRY ZONE",
                (dx1, dy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    # Run tracking continuously
    results = model.track(frame, conf=0.5, classes=[0], persist=True, verbose=False)

    ids = []
    boxes = []

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

    valid_ids = set()

    # Draw bounding boxes + filter to entry zone
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        if dx1 < cx < dx2 and dy1 < cy < dy2:
            valid_ids.add(track_id)

    # -------------------------------
    # START SCAN WHEN API CALLED
    # -------------------------------
    

    if scan_triggered and not scanning:
        scanning = True
        scan_start = time.time()
        flagged = False
        primary_id = None
        secondary_detect_time = None
        scan_triggered = False
        print("Verification triggered by biometric API")

    # -------------------------------
    # BEHAVIOR LOGIC DURING WINDOW
    # -------------------------------
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

        cv2.putText(frame,
                    f"Verifying entry... {round(3 - elapsed, 1)}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        # End 3-second verification
        if elapsed > 3:
            scanning = False
            print("Verification complete")

            if flagged:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"clips/event_{ts}.mp4"

                h, w, _ = frame.shape
                out = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS,
                    (w, h)
                )

                for f in buffer:
                    out.write(f)

                out.release()
                print("⚠ Unauthorized entry recorded")
            else:
                print("Entry verified successfully")

    # -------------------------------
    # DISPLAY FINAL STATUS
    # -------------------------------
    if not scanning and scan_start is not None:
        status = (
            "UNAUTHORIZED ENTRY DETECTED"
            if flagged else
            "ACCESS VERIFIED"
        )

        color = (0, 0, 255) if flagged else (0, 200, 0)

        cv2.putText(frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    color,
                    3)

    cv2.imshow("Axentry Access Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
