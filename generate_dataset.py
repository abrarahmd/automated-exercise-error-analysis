import os
import shutil
import csv

# Separate Mistakes and Perfect folder and cut-paste mistakes videos and Perfect video in the respected directory
mistake_dir = "trimmed_videos/mistakes" 
perfect_dir = "trimmed_videos/perfect"

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)


perfect_clips = [f for f in os.listdir(perfect_dir) if f.endswith(".mp4")]
if not perfect_clips:
    raise Exception("No perfect clip found.")

perfect_clip = perfect_clips[0]
perfect_src_path = os.path.join(perfect_dir, perfect_clip)
perfect_dst_path = os.path.join(dataset_dir, perfect_clip)
shutil.copy2(perfect_src_path, perfect_dst_path)


rows = []
mistake_clips = [f for f in os.listdir(mistake_dir) if f.endswith(".mp4")]
for mc in mistake_clips:
    label = "_".join(mc.split("_")[:-1])
    mc_src_path = os.path.join(mistake_dir, mc)
    mc_dst_path = os.path.join(dataset_dir, mc)
    shutil.copy2(mc_src_path, mc_dst_path)
    rows.append((mc, perfect_clip, label))

csv_path = os.path.join(dataset_dir, "dataset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mistake_clip", "perfect_clip", "label"])
    writer.writerows(rows)

print(f"Dataset created in folder: {dataset_dir}")
