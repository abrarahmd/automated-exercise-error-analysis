import cv2
import os

video_folder = "reduced_480p"  # your video folder
video_files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]

for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_name}")
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_name}: {frame_count} frames")
    cap.release()
