import os
import ffmpeg

input_folder = "yt_clips_downloaded"
output_folder = "yt_clips_trimmed"
os.makedirs(output_folder, exist_ok=True)

trim_segments = {
    "ytv_downloaded.mp4": [('34', '38'), ('59', '61'), ('67', '69'), ('173', '176')],
}

for video_name, segments in trim_segments.items():
    input_path = os.path.join(input_folder, video_name)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    for i, (start_str, end_str) in enumerate(segments):

        start = float(start_str)
        end = float(end_str)
        duration = end - start

        out_name = f"{os.path.splitext(video_name)[0]}_part{i+1}.mp4"
        out_path = os.path.join(output_folder, out_name)

        try:
            print(f"Trimming {video_name} [{start}s to {end}s] -> {out_name}")
            (
                ffmpeg.input(input_path, ss=start, t=duration) \
                .output(out_path, codec='libx264', preset='fast', movflags='faststart') \
                .run(cmd=r"D:\Automated Exercise Error Analysis\ffmpeg\bin\ffmpeg.exe", overwrite_output=True)

            )
        except ffmpeg.Error as e:
            print(f"Error trimming segment {i+1}: {e.stderr.decode()}")
