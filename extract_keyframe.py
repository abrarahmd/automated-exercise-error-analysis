import cv2
import os

def extract_keyframes(video_path, output_dir, num_frames=3, prefix="frame"):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    extracted = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_filename = os.path.join(output_dir, f"{prefix}_{extracted+1}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted += 1
    cap.release()
    print(f"Extracted {extracted} keyframes from {video_path}")


extract_keyframes("reference/proper_crunches.mp4", "reference", num_frames=3, prefix="trainer")
extract_keyframes("user_videos/pulling_neck.mp4", "user_videos", num_frames=3, prefix="userFirst")
extract_keyframes("user_videos/tucking_chin.mp4", "user_videos", num_frames=3, prefix="userSecond")
extract_keyframes("user_videos/using_momentum.mp4", "user_videos", num_frames=3, prefix="userThird")

