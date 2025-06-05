from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import time
from flask import Flask, Response
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLOv8 and MediaPipe
model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture and output
cap = cv2.VideoCapture("input_video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Global frame for streaming
current_frame = None
frame_lock = threading.Lock()

# Alert log
def log_alert(person_id, event_type, confidence, timestamp):
    with open("alerts.log", "a") as f:
        f.write(f"{timestamp}: Person {person_id} - {event_type} (Confidence: {confidence})\n")

# Tracking dictionary
person_tracker = {}
person_id = 0

def process_video():
    global current_frame, person_tracker, person_id
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame_rgb)

        # Run YOLOv8 inference
        results_yolo = model(frame, classes=[0], conf=0.5)

        # Process YOLO detections
        current_persons = []
        for r in results_yolo:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_persons.append((center, (x1, y1, x2, y2)))

        # Update tracking
        new_tracker = {}
        for center, bbox in current_persons:
            matched = False
            for pid, (last_center, last_bbox) in person_tracker.items():
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                if dist < 50:
                    new_tracker[pid] = (center, bbox)
                    matched = True
                    break
            if not matched:
                new_tracker[person_id] = (center, bbox)
                person_id += 1

        person_tracker = new_tracker

        # Process cigarette detection (hand-to-mouth gesture)
        for pid, (center, (x1, y1, x2, y2)) in person_tracker.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Define mouth region
            mouth_region = (x1, y1, x2, y1 + (y2 - y1) // 3)
            cigarette_detected = False

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                    if (mouth_region[0] < wrist_x < mouth_region[2] and
                        mouth_region[1] < wrist_y < mouth_region[3]):
                        cigarette_detected = True
                        cv2.putText(frame, "Cigarette Detected!", (x1, y1 - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        log_alert(pid, "Cigarette", 0.9, time.strftime("%Y-%m-%d %H:%M:%S"))
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save frame to output video
        out.write(frame)

        # Update current frame for streaming
        with frame_lock:
            ret, buffer = cv2.imencode('.jpg', frame)
            current_frame = buffer.tobytes()

        # Short sleep to prevent CPU overload
        time.sleep(0.01)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Monitoring Demo</title>
    </head>
    <body>
        <h1>Real-Time CCTV Monitoring Demo</h1>
        <img src="/video_feed" style="width:640px;">
        <p>Check <code>output_video.mp4</code> and <code>alerts.log</code> for results.</p>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
            time.sleep(0.03)  # Control frame rate
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video processing in a separate thread
    threading.Thread(target=process_video, daemon=True).start()
    # Run Flask app
    app.run(host='0.0.0.0', port=8000)