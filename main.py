import os
import glob
import shutil
from feedback import extract_keypoints, compare_keypoints, summarize_mistakes

os.makedirs("keypoints", exist_ok=True)

source_folder = "yt_clips_text_removed"
reference_folder = "reference"
user_video_folder = "user_videos"

os.makedirs(reference_folder, exist_ok=True)
os.makedirs(user_video_folder, exist_ok=True)

videos = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]
for video in videos:
    src_path = os.path.join(source_folder, video)
    if video == "proper.mp4":
        dst_path = os.path.join(reference_folder, video)
    else:
        dst_path = os.path.join(user_video_folder, video)
    shutil.copy(src_path, dst_path)
    print(f"Moved {video} to {dst_path}")

reference_video = os.path.join(reference_folder, "proper.mp4")
reference_keypoints = os.path.join("keypoints", "reference_proper.npy")

if not os.path.exists(reference_keypoints):
    extract_keypoints(reference_video, reference_keypoints)

user_videos = glob.glob(os.path.join(user_video_folder, "*.mp4"))
print(f"Found user videos: {user_videos}")

for user_video in user_videos:
    user_filename = os.path.basename(user_video)
    user_name = os.path.splitext(user_filename)[0]

    user_key_path = os.path.join("keypoints", f"{user_name}.npy")

    extract_keypoints(user_video, user_key_path)

    errors = compare_keypoints(reference_keypoints, user_key_path)

    print(f"\nMistakes in {user_name}:")
    summary = summarize_mistakes(errors, total_frames=len(errors), min_frame_threshold=5)
    for line in summary:
        print(line)