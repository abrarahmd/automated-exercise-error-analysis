# Automated Exercise Error Analysis — WebSocket Implementation

This project implements a **basic real-time pose keypoint extraction system** using:
- `socket.io` for real-time communication between browser and server
- `MediaPipe` for extracting pose keypoints
- `OpenCV` for drawing keypoints and saving skeleton frames

✅ **User privacy friendly:**  
📷 Webcam video frames are processed in real-time at 5 FPS to extract pose keypoints.  
🚫 No user video is saved — only keypoints (`.npy`) and skeleton frames (`.jpg`) are stored.

---

## 📁 Project Structure
```bash
websocket/
├── client.js # Client-side logic: webcam stream, frame sender
├── index.html # Client-side UI
├── server.py # Socket.IO server: keypoint extraction & storage
├── saved_keypoints/ # Directory to store extracted keypoints
├── saved_frames/ # Directory to store skeleton frames
```

---

## 🖥️ Features
- Captures webcam video on the browser
- Sends frames to Python server at **5 FPS**
- Server extracts **17 selected pose keypoints**
- Saves:
  - Keypoints (`.npy`) in `saved_keypoints/`
  - Skeleton-only frames (`.jpg`) in `saved_frames/`
- Sends feedback to browser after each frame is processed

---

## 🛠️ Technologies
- Frontend:
  - HTML5 + `<video>` + `<canvas>`
  - `socket.io-client`
- Backend:
  - `python-socketio`
  - `mediapipe`
  - `opencv-python`
  - `werkzeug`

---

## 🚀 Getting Started

### 🔗 Clone the repository
```bash
git clone https://github.com/abrarahmd/automated-exercise-error-analysis.git
cd automated-exercise-error-analysis
git checkout -b websocket_implementation
```

---

### 🐍 Install Python Dependencies
```bash
pip install python-socketio werkzeug mediapipe opencv-python numpy
```

---

### ▶️ Run the Server

```bash
cd websocket
python server.py
```
Then open index.html.

Cheers!
