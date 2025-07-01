import openai
import base64
import requests
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
            "You will be shown six images:\n"
            "Trainer Image 1\n"
            "Trainer Image 2\n"
            "Trainer Image 3\n"
            "Learner Image 1\n"
            "Learner Image 2\n"
            "Learner Image 3\n\n"
            "Compare the learner's posture and movement in each frame with the trainer's. "
            "List the mistakes or differences in form, balance, or position. "
            "Provide feedback in numbered points."
        ),
    }
]

for i, img_b64 in enumerate(trainer_imgs):
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_b64}",
            "detail": f"Trainer Image {i+1}"
        }
    })

for i, img_b64 in enumerate(learner_imgs):
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_b64}",
            "detail": f"Learner Image {i+1}"
        }
    })



for i in range(3):
    response = openai.chat.completions.create(
        model="qwen2.5vl:latest",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Compare the following two images:\n"
                        f"Image A (Trainer Frame {i+1}) and Image B (Learner Frame {i+1}).\n"
                        "Identify any mistakes or posture differences."
                    )
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{trainer_imgs[i]}" }},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{learner_imgs[i]}" }},
            ],
        }],
        max_tokens=1024,
    )
    print(f"\n💬 Comparison for Frame {i+1}:\n", response.choices[0].message.content)
