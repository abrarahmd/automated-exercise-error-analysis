import socketio
from werkzeug.serving import run_simple
import numpy as np
import cv2
import base64
import mediapipe as mp
import os
import time

# Create Socketio server and WSGI application
# This will handle real-time communication between client and server
sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
app = socketio.WSGIApp(sio)

# Initialize MediaPipe Pose
# This will be used to process incoming video frames and extract keypoints
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Define the selected joints to extract
# These are the indices of the joints we want to keep from the full set of 33
SELECTED_JOINTS = [
    0, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30
]

# Create a mapping from original indices to selected indices
# This will help us filter the connections to only those involving selected joints
index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(SELECTED_JOINTS)}
filtered_connections = []

# Filter the pose connections to only include those that involve selected joints
# This will create a list of tuples where each tuple contains the new indices of the connected joints
for (start, end) in mp_pose.POSE_CONNECTIONS:
    # Only keep connections where both joints are in the selected set
    # This ensures we only draw lines between the joints we care about
    if start in index_map and end in index_map:
        filtered_connections.append((index_map[start], index_map[end]))

# Create directories to save keypoints and frames
# This will store the extracted keypoints and corresponding frames for later analysis
os.makedirs("saved_keypoints", exist_ok=True)
os.makedirs("saved_frames", exist_ok=True)

# Define SocketIO events
# These functions will handle incoming connections, disconnections, and frame processing
@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.on('frame')
def handle_frame(sid, data):
    print(f"Received frame from {sid}")

    # Decode the base64-encoded JPEG image
    # This will convert the incoming data into a format we can process with OpenCV
    jpg_original = base64.b64decode(data)
    # Convert the JPEG image to a NumPy array
    # This allows us to manipulate the image using OpenCV functions
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # Decode the NumPy array into an OpenCV image
    # This will convert the raw byte data into a format we can work with
    frame = cv2.imdecode(jpg_as_np, flags=1)

    # Process the image with MediaPipe Pose
    # This will extract the pose landmarks from the image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the image from BGR to RGB format as required by MediaPipe
    # MediaPipe works with RGB images, so we need to convert the BGR image from OpenCV
    results = pose.process(image_rgb)

    # Initialize keypoints array
    # This will hold the coordinates and visibility of the selected joints
    keypoints = np.zeros((len(SELECTED_JOINTS), 4))

    # Get the height and width of the frame
    # This will be used to scale the keypoints correctly
    h, w = frame.shape[:2]

    if results.pose_landmarks:
        # If landmarks are detected, extract the keypoints
        # This will create an array of coordinates for the selected joints
        landmarks = results.pose_landmarks.landmark
        # Extract the coordinates (x, y, z) and visibility for each selected joint
        # The visibility indicates how confident MediaPipe is about the joint's presence
        keypoints = np.array([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z, landmarks[i].visibility]
            for i in SELECTED_JOINTS
        ])

        # Scale the keypoints to the frame size
        # This will convert the normalized coordinates (0 to 1) to pixel coordinates
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Create a black frame to draw the keypoints and skeleton
        for idx, (x, y, z, v) in enumerate(keypoints):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        # Draw the keypoints on the frame
        for start_idx, end_idx in filtered_connections:
            x1, y1, *_ = keypoints[start_idx]
            x2, y2, *_ = keypoints[end_idx]
            p1 = int(x1 * w), int(y1 * h)
            p2 = int(x2 * w), int(y2 * h)
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        print("✅ Keypoints extracted & skeleton drawn.")

    else:
        # If no landmarks are detected, create a black frame
        # This will be used to indicate that no pose was detected in the frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        print("No landmarks detected, saved black image.")

    # Encode the processed frame back to JPEG
    # This will convert the OpenCV image back to a format suitable for transmission
    timestamp = int(time.time() * 1000)
    np.save(os.path.join("saved_keypoints", f"{timestamp}.npy"), keypoints)
    # Save the keypoints as a NumPy array for later analysis
    # This will allow us to easily load and process the keypoints later
    cv2.imwrite(os.path.join("saved_frames", f"{timestamp}.jpg"), frame)
    # Save the processed frame as a JPEG image
    # This will allow us to visualize the keypoints and skeleton drawn on the frame

    
    sio.emit('feedback', 'Keypoints & skeleton saved', room=sid)
    # Send feedback to the client indicating that the keypoints and skeleton were saved

if __name__ == '__main__':
    print("🚀 Server running at http://0.0.0.0:5000")
    run_simple('0.0.0.0', 5000, app, threaded=True)
