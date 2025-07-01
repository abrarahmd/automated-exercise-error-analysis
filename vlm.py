import openai
import base64
from io import BytesIO
from PIL import Image

def encode_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


trainer_imgs = [
    encode_image("reference/trainer_1.jpg"),
    encode_image("reference/trainer_2.jpg"),
    encode_image("reference/trainer_3.jpg"),
]

learner_imgs = [
    encode_image("user_videos/userFirst_1.jpg"),
    encode_image("user_videos/userFirst_2.jpg"),
    encode_image("user_videos/userFirst_3.jpg"),
]

openai.base_url = "http://172.16.4.134:11434/v1/"
openai.api_key = 'ollama'


content = [
    {
        "type": "text",
        "text": (
            "You are a certified fitness trainer evaluating a user's sit-up technique.\n\n"
            "You will be shown six labeled images:\n"
            "Image 1: Trainer Start\n"
            "Image 2: Trainer Middle\n"
            "Image 3: Trainer End\n"
            "Image 4: Learner Start\n"
            "Image 5: Learner Middle\n"
            "Image 6: Learner End\n\n"
            "Compare the learner's form and movement to the trainer's across the sequence.\n"
            "Identify any differences in posture, alignment, balance, or movement quality.\n"
            "Highlight specific mistakes — such as pulling on the neck, using momentum, or misalignments — that may impact performance or risk injury.\n"
            "Give clear feedback in numbered points, connected across all frames."
        ),
    }
]


trainer_labels = ["Trainer Start", "Trainer Middle", "Trainer End"]
for i, (img_b64, label) in enumerate(zip(trainer_imgs, trainer_labels)):
    content.append({"type": "text", "text": f"Image {i+1}: {label}"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
    })


learner_labels = ["Learner Start", "Learner Middle", "Learner End"]
for i, (img_b64, label) in enumerate(zip(learner_imgs, learner_labels), start=4):
    content.append({"type": "text", "text": f"Image {i}: {label}"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
    })


response = openai.chat.completions.create(
    model="qwen2.5vl:latest",
    messages=[{
        "role": "user",
        "content": content,
    }],
    max_tokens=1024,
    temperature=0.1,
    top_p=0.001,
)

print("📋 Overall Feedback:\n", response.choices[0].message.content)
