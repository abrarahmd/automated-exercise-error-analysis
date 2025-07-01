import cv2
import os

input_path = "visualizations/using_momentum_5fps_vis.mp4"
output_path = "reduced_480p/using_momentum_480p.mp4"
target_resolution = (854, 480) 

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, target_resolution)
    out.write(resized)

cap.release()
out.release()
print(f"✅ Saved 480p video to: {output_path}")
