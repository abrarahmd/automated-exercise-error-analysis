from dotenv import load_dotenv
import os
import glob
import openai
import shutil
import traceback
from feedback import extract_keypoints, compare_keypoints, summarize_mistakes

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1"

def get_llm_feedback(reference_name, error_summary):
    if not error_summary:
        return "No significant form issues detected."
    
    prompt = f"""You are a professional fitness coach helping people fix their form during exercises.

The user performed the workout: {reference_name}

The pose analyzer detected these joint-level issues over multiple frames:
{chr(10).join(error_summary)}

Your task is:
1. Summarize these technical errors into 2–3 high-level, human-friendly feedback points.
2. Group related issues (like knees and ankles) into a single point if possible.
3. Focus on functional mistakes, e.g., "neck pulling", "feet not planted", "elbows flaring", etc.

Format your answer with:
- A short mistake name
- A clear one-sentence fix

Example:
Neck pulling — Avoid pulling with your head; keep your neck relaxed.

Now write only the list of summarized feedback below:
"""

    try:
        client = openai.OpenAI(
            api_key=openai.api_key,
            base_url=openai.base_url
        )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a concise fitness coach. Group and summarize body part issues into meaningful form feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            stop=["\n\n"]
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in LLM API call: {e}")
        return "Unable to generate feedback due to API error."


os.makedirs("keypoints", exist_ok=True)
source_folder = "yt_clips_text_removed"
reference_folder = "reference"
user_video_folder = "user_videos"

os.makedirs(reference_folder, exist_ok=True)
os.makedirs(user_video_folder, exist_ok=True)

if not os.path.exists(source_folder):
    print(f"Error: Source folder '{source_folder}' not found!")
    exit(1)

videos = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]

for video in videos:
    src_path = os.path.join(source_folder, video)
    if video == "proper_crunches_5fps.mp4":
        dst_path = os.path.join(reference_folder, video)
    else:
        dst_path = os.path.join(user_video_folder, video)
    
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error moving {video}: {e}")

reference_video = os.path.join(reference_folder, "proper_crunches_5fps.mp4")
reference_keypoints = os.path.join("keypoints", "reference_proper_5fps.npy")

if not os.path.exists(reference_video):
    print("Error: Reference video 'proper_crunches_5fps.mp4' not found!")
    exit(1)

if not os.path.exists(reference_keypoints):
    try:
        extract_keypoints(reference_video, reference_keypoints)
    except Exception as e:
        print(f"Error extracting reference keypoints: {e}")
        exit(1)

user_videos = glob.glob(os.path.join(user_video_folder, "*.mp4"))

for user_video in user_videos:
    user_filename = os.path.basename(user_video)
    user_name = os.path.splitext(user_filename)[0]
    
    print(f"\n{'='*50}")
    print(f"Analyzing: {user_name}")
    print(f"{'='*50}")
    
    user_key_path = os.path.join("keypoints", f"{user_name}.npy")
    
    try:
        extract_keypoints(user_video, user_key_path)
        errors = compare_keypoints(reference_keypoints, user_key_path)
        summary = summarize_mistakes(errors, total_frames=len(errors), min_frame_threshold=5)
        print(summary)

        if summary:

            feedback = get_llm_feedback("proper_crunches_5fps", summary)
            print("\nAI Workout Feedback:")
            print("-" * 40)
            print(feedback)
        else:
            print("No significant mistakes detected!")
            
    except Exception as e:
        print(f"Error processing {user_name}:")
        traceback.print_exc()