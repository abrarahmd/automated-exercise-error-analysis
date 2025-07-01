import numpy as np
import cv2
import mediapipe as mp
import os

# Your selected joints
SELECTED_JOINTS = [
    0, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30
]

# Mapping original pose indices → indices in your subset
index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(SELECTED_JOINTS)}

# Load MediaPipe pose connections and filter for only selected joints
mp_pose = mp.solutions.pose
filtered_connections = []
for (start, end) in mp_pose.POSE_CONNECTIONS:
    if start in index_map and end in index_map:
        filtered_connections.append((index_map[start], index_map[end]))

keypoints_dir = "keypoints"
frame_size = (640, 480)
fps = 5
os.makedirs("visualizations", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

files = sorted([f for f in os.listdir(keypoints_dir) if f.endswith((".npy", ".npz"))])

def draw_skeleton(frame_kp):
    img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    # Draw connections
    for start_idx, end_idx in filtered_connections:
        start = frame_kp[start_idx]
        end = frame_kp[end_idx]

        if not np.any(np.isnan(start[:2])) and not np.any(np.isnan(end[:2])):
            x1, y1 = int(start[0] * frame_size[0]), int(start[1] * frame_size[1])
            x2, y2 = int(end[0] * frame_size[0]), int(end[1] * frame_size[1])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    for kp in frame_kp:
        x, y = kp[0], kp[1]
        if not np.isnan(x) and not np.isnan(y):
            px, py = int(x * frame_size[0]), int(y * frame_size[1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)  # red points
    return img

for file in files:
    path = os.path.join(keypoints_dir, file)
    if file.endswith(".npz"):
        data = np.load(path)
        keypoints = data["keypoints"]  # shape (frames, joints, 4)
    else:
        keypoints = np.load(path)  # shape (frames, joints, 4)

    output_video = os.path.join("visualizations", f"{os.path.splitext(file)[0]}_vis.mp4")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    if keypoints.ndim == 3:
        for frame_kp in keypoints:
            img = draw_skeleton(frame_kp)
            video_writer.write(img)
    elif keypoints.ndim == 2:
        img = draw_skeleton(keypoints)
        video_writer.write(img)
    else:
        print(f"Unexpected keypoints shape: {keypoints.shape} in file {file}")

    video_writer.release()
    print(f"✅ Saved visualization video: {output_video}")
