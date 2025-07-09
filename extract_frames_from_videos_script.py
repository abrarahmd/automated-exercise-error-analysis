import cv2
import os

def extract_frames_from_videos(
    input_dir="final_videos",
    output_dir="final_frames",
    video_files=None
):
    if video_files is None:
        video_files = [
            "pulling_neck_480p_5fps_vis.mp4",
            "proper_crunches_480p_5fps_vis.mp4",
            "tucking_chin_480p_5fps_vis.mp4",
            "using_momentum_480p_5fps_vis.mp4"
        ]

    os.makedirs(output_dir, exist_ok=True)

    for video_name in video_files:
        video_path = os.path.join(input_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open {video_name}")
            continue

        video_base = os.path.splitext(video_name)[0]
        frame_output_path = os.path.join(output_dir, video_base)
        os.makedirs(frame_output_path, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(frame_output_path, f"frame_{frame_idx:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1

        cap.release()
