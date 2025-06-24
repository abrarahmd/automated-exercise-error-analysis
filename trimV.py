import os
import ffmpeg

input_folder = "clips"
output_folder = "trimmed_videos"
os.makedirs(output_folder, exist_ok=True)

trim_segments = {
    "Crunches.mp4": [(34, 38), (58, 60), (65, 67), (173, 181)],
}

for video_name, segments in trim_segments.items():
    input_path = os.path.join(input_folder, video_name)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    for i, (start, end) in enumerate(segments):
        duration = end - start
        out_name = f"{os.path.splitext(video_name)[0]}_part{i+1}.mp4"
        out_path = os.path.join(output_folder, out_name)

        try:
            print(f"Trimming {video_name} [{start}s to {end}s] -> {out_name}")
            (
                ffmpeg
                .input(input_path, ss=start, t=duration)
                .output(out_path, codec='libx264', preset='fast', movflags='faststart')
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print(f"Error trimming segment {i+1}: {e.stderr.decode()}")
