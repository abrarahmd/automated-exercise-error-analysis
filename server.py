import socketio
from werkzeug.serving import run_simple
import numpy as np
import cv2
import base64
import mediapipe as mp
import os
import time

# Create Socketio server and WSGI application. This will handle real-time communication between client and server.
sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
app = socketio.WSGIApp(sio)

# Initialize MediaPipe Pose. This will be used to process incoming video frames and extract keypoints.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Define the selected joints to extract, These are the indices of the joints we want to keep from the full set of 33.
SELECTED_JOINTS = [
    0, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30
]

# Filter the connections to only those involving selected joints
index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(SELECTED_JOINTS)}
filtered_connections = []
for (start, end) in mp_pose.POSE_CONNECTIONS:
    if start in index_map and end in index_map:
        filtered_connections.append((index_map[start], index_map[end]))

# Create output directory if it doesn’t already exist
os.makedirs("user_frames", exist_ok=True)

# Event: when a client connects
@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")

# Event: when a client disconnects
@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Event: when a frame is received from a client
@sio.on('frame')
def handle_frame(sid, data):
    print(f"Received frame from {sid}")
    
    # Decode base64 JPEG data into OpenCV image
    jpg_original = base64.b64decode(data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Initialize keypoints array with zeros
    keypoints = np.zeros((len(SELECTED_JOINTS), 4))

    # Get frame dimensions
    h, w = frame.shape[:2]

    if results.pose_landmarks: # Extract keypoints if detected
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z, landmarks[i].visibility]
            for i in SELECTED_JOINTS
        ])
        # Draw skeleton on a black frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw keypoints as red circles
        for idx, (x, y, z, v) in enumerate(keypoints):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
         # Draw skeleton connections as green lines
        for start_idx, end_idx in filtered_connections:
            x1, y1, *_ = keypoints[start_idx]
            x2, y2, *_ = keypoints[end_idx]
            p1 = int(x1 * w), int(y1 * h)
            p2 = int(x2 * w), int(y2 * h)
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        print("Keypoints extracted & skeleton drawn.")

    else:
        # If no landmarks detected, save a blank black image
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        print("No landmarks detected, saved black image.")

    # Save frame with timestamp
    timestamp = int(time.time() * 1000)
    cv2.imwrite(os.path.join("user_frames", f"{timestamp}.jpg"), frame)

if __name__ == '__main__':
    print("🚀 Server running at http://0.0.0.0:5000")
    run_simple('0.0.0.0', 5000, app, threaded=True)
