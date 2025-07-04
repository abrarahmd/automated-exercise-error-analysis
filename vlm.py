import openai
import base64
from io import BytesIO
from PIL import Image
import os
import glob
import math
import cv2

openai.base_url = "http://172.16.7.83:11434/v1/"
openai.api_key = 'ollama'
model_name = "qwen2.5vl:32b"

reference_dir = "frames_all/reference_proper_480p"
user_dirs = [
    "frames_all/pulling_neck_480p",
    "frames_all/tucking_chin_480p",
    "frames_all/using_momentum_480p"
]

OUTPUT_DIR = "llm_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_segments(frames, segment_count=3):
    total = len(frames)
    seg_len = math.ceil(total / segment_count)
    return [frames[i*seg_len:(i+1)*seg_len] for i in range(segment_count)]

def extract_segment_b64(image_paths):
    return [encode_image(p) for p in image_paths]

def generate_prompt():
    return {
        "type": "text",
        "text": (
            "You are a certified fitness trainer evaluating different kind of workout. Based on the reference frames, figure it out what kind of workout is this.\n\n"
            "Each comparison contains 3 reference frames (proper workout) and 3 user frames.\n\n"
            "Reference: Start, Middle, End\n"
            "User: Start, Middle, End\n\n"
            "Compare them. Mention a detailed comparison.\n"
            "Generate user instructions.\n"
            "Focus on overall patterns like pulling neck, tucking chin, using momentum, poor alignment.\n"
            "Be concise. Format like: 'Mistake: Explanation'\n"
            "Word counts should be around 100-200 words.\n"

        )
    }

def run_llm(reference_b64s, user_b64s):
    content = [generate_prompt()]

    for i, b64 in enumerate(reference_b64s):
        content.append({"type": "text", "text": f"Reference Frame {i+1}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    for i, b64 in enumerate(user_b64s):
        content.append({"type": "text", "text": f"User Frame {i+1}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.2,
    )

    return response.choices[0].message.content

def frames_to_video(frames, save_path, fps=5):
    if not frames:
        return
    h, w = cv2.imread(frames[0]).shape[:2]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame_path in frames:
        img = cv2.imread(frame_path)
        out.write(img)
    out.release()

reference_frames = sorted(glob.glob(os.path.join(reference_dir, "*.jpg")))
ref_segments = get_segments(reference_frames, segment_count=3)
ref_b64_segments = [extract_segment_b64(seg) for seg in ref_segments]

for user_dir in user_dirs:
    user_name = os.path.basename(user_dir)
    user_frames = sorted(glob.glob(os.path.join(user_dir, "*.jpg")))
    user_segments = get_segments(user_frames, segment_count=3)
    user_b64_segments = [extract_segment_b64(seg) for seg in user_segments]

    reference_set = sum(ref_b64_segments, [])[:3]
    user_set = sum(user_b64_segments, [])[:3]

    print(f"\nAnalyzing: {user_name}")
    feedback = run_llm(reference_set, user_set)

    with open(os.path.join(OUTPUT_DIR, f"{user_name}_feedback.txt"), "w") as f:
        f.write(feedback)

    print("Feedback saved.")
