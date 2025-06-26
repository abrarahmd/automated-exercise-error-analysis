import cv2
import easyocr
import numpy as np
import os
import glob

reader = easyocr.Reader(['en'])

input_folder = "yt_clips_trimmed"  
output_folder = "yt_clips_text_removed"
os.makedirs(output_folder, exist_ok=True)

# Get all video files in input folder (adjust extensions as needed)
video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

for input_path in video_files:
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f"{filename}")

    print(f"Processing {filename}...")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = reader.readtext(frame)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (bbox, text, prob) in results:
            pts = np.array(bbox).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        inpainted = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        out.write(inpainted)

        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")
        frame_count += 1

    cap.release()
    out.release()

    print(f"Done processing {filename}. Saved to {output_path}\n")
