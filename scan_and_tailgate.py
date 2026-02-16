from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
BUFFER_SECONDS = 10
buffer = deque(maxlen=FPS * BUFFER_SECONDS)

scanning = False
scan_start = None
flagged = False

primary_id = None
secondary_detect_time = None
TOLERANCE_SECONDS = 1.0

os.makedirs("clips", exist_ok=True)

print("Press 'S' to simulate scan | Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    buffer.append(frame.copy())
    key = cv2.waitKey(1) & 0xFF

    # Start scan
    if key == ord("s") and not scanning:
        scanning = True
        scan_start = time.time()
        flagged = False
        primary_id = None
        secondary_detect_time = None
        print("SCAN STARTED")

    if scanning:
        elapsed = time.time() - scan_start

        # Run tracking
        results = model.track(frame, conf=0.5, classes=[0], persist=True)

        ids = []
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().tolist()

        current_ids = set(ids)

        # Identify primary entrant
        if primary_id is None and len(current_ids) >= 1:
            primary_id = list(current_ids)[0]

        # Check for secondary entrant
        if primary_id is not None:
            secondary_ids = current_ids - {primary_id}

            if len(secondary_ids) > 0:
                if secondary_detect_time is None:
                    secondary_detect_time = time.time()

                elif time.time() - secondary_detect_time >= TOLERANCE_SECONDS:
                    flagged = True
            else:
                secondary_detect_time = None

        cv2.putText(frame, f"SCANNING... {round(3-elapsed,1)}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(frame, f"Tracked IDs: {list(current_ids)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if elapsed > 3:
            scanning = False
            print("SCAN ENDED")

            if flagged:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"clips/flagged_{ts}.mp4"

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
                print(f"🚨 TAILGATING DETECTED — clip saved: {path}")
            else:
                print("✅ OK — no tailgating detected")

    # Status display
    if not scanning and scan_start is not None:
        text = "TAILGATING ❌" if flagged else "OK ✅"
        color = (0,0,255) if flagged else (0,255,0)
        cv2.putText(frame, text,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Axentry Behavior-Level Pilot", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
