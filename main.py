from dotenv import load_dotenv
import subprocess
import os
import glob
import shutil
import traceback
from feedback import extract_keypoints, compare_keypoints, summarize_mistakes

# Convert Videos -> 480p
SRC_DIR = "clips"
OUT_DIR = os.path.join(SRC_DIR, "converted")
os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if not fname.lower().endswith(".mp4"):
        continue

    in_path = os.path.join(SRC_DIR, fname)
    base, _ = os.path.splitext(fname)
    out_path = os.path.join(OUT_DIR, f"{base}_480p_5fps.mp4")

    cmd = [
        "ffmpeg",
        "-i", in_path,
        "-vf", "scale=-2:480",   # Resize Videos -> 480p
        "-r", "5",               # Change Frame Rate -> 5 fps
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR processing {fname}: {e}")
# End

# Move Videos to Desired Folders
load_dotenv()
source_folder = "clips/converted"
reference_folder = "reference"
user_video_folder = "user_videos"
keypoint_folder = "keypoints"

os.makedirs(reference_folder, exist_ok=True)
os.makedirs(user_video_folder, exist_ok=True)
os.makedirs(keypoint_folder, exist_ok=True)

if not os.path.exists(source_folder):
    print(f"Error: Source folder '{source_folder}' not found!")
    exit(1)

videos = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]

for video in videos:
    src_path = os.path.join(source_folder, video)
    if video == "Proper_Crunches_480p_5fps.mp4":
        dst_path = os.path.join(reference_folder, video)
    else:
        dst_path = os.path.join(user_video_folder, video)
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying {video}: {e}")
# End
        
reference_video = os.path.join(reference_folder, "Proper_Crunches_480p_5fps.mp4")
user_videos = glob.glob(os.path.join(user_video_folder, "*.mp4"))
reference_keypoints = os.path.join("keypoints", "Proper_Crunches_480p_5fps.npy")


if not os.path.exists(reference_video):
    print("Error: Reference video 'proper_crunches_5fps.mp4' not found!")
    exit(1)

# Extract Keypoints
if not os.path.exists(reference_keypoints):
    try:
        extract_keypoints(reference_video, reference_keypoints)
    except Exception as e:
        print(f"Error extracting reference keypoints: {e}")
        exit(1)

for user_video in user_videos:
    user_filename = os.path.basename(user_video)
    user_name = os.path.splitext(user_filename)[0]

    user_key_path = os.path.join("keypoints", f"{user_name}.npy")

    try:
        extract_keypoints(user_video, user_key_path)
            
    except Exception as e:
        print(f"Error processing {user_name}:")
        traceback.print_exc()