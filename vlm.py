import base64
from pathlib import Path
import requests


def encode_images_to_base64(folder_path):
    base64_images = {}
    for image_path in sorted(Path(folder_path).glob("*.jpg")):
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            base64_images[image_path.stem] = encoded_string
    return base64_images




def build_multi_frame_message(trainer_images, learner_images, max_frames=3):
    shared_keys = sorted(set(trainer_images) & set(learner_images))
    shared_keys = shared_keys[:max_frames]  # Limit to avoid token overflow

    content = [
        {
            "type": "text",
            "text": f"Compare the learner's pose to the trainer's pose in the following {len(shared_keys)} frames. Identify any mistakes in the learner’s form and explain clearly what is wrong in each frame."
        }
    ]

    for key in shared_keys:
        content.extend([
            {"type": "text", "text": f"🖼 Frame {key}:"},
            {"type": "text", "text": "Trainer:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{trainer_images[key]}" }},
            {"type": "text", "text": "Learner:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{learner_images[key]}" }},
        ])

    return [{
        "role": "user",
        "content": content
    }]


# Load images
trainer_images = encode_images_to_base64("reference")
learner_images = encode_images_to_base64("user_videos")

# Build messages
messages = build_multi_frame_message(trainer_images, learner_images, max_frames=3)

# API endpoint and headers
api = "http://172.16.4.134:11434/v1/"
api_key = "ollama"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

# Payload with built messages
payload = {
    "model": "hf.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M",
    "messages": messages,
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.95,
}

response = requests.post(api, headers=headers, json=payload)
print(response.json() if response.ok else response.text)
