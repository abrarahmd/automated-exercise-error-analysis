from dotenv import load_dotenv
import os
import glob
import openai
import shutil
import traceback
from feedback import extract_keypoints, compare_keypoints, summarize_mistakes

# Use environment variable for API key (more secure)
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1"

def get_llm_feedback(reference_name, error_summary):
    prompt = f"""I am a fitness coach. I am analyzing exercise posture based on a reference video: {reference_name}. 
Here are the mistakes detected in the user's performance:

{chr(10).join(error_summary)}

Based on this, identify what kind of workout form mistakes the person is making. Avoid the mistakes that does not align with the 
workout reference video type. Be specific (e.g., "arm too low", "neck strain", "poor core engagement"), 
and give correction advice for each that is relevant to the reference video."""

    try:
        client = openai.OpenAI(
            api_key=openai.api_key,
            base_url=openai.base_url
        )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a fitness coach analyzing posture data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in LLM API call: {e}")
        return "Unable to generate feedback due to API error."

# Create necessary directories
os.makedirs("keypoints", exist_ok=True)

# Define folder paths
source_folder = "yt_clips_text_removed"
reference_folder = "reference"
user_video_folder = "user_videos"

# Create directories if they don't exist
os.makedirs(reference_folder, exist_ok=True)
os.makedirs(user_video_folder, exist_ok=True)

# Check if source folder exists
if not os.path.exists(source_folder):
    print(f"Error: Source folder '{source_folder}' not found!")
    exit(1)

# Sort videos into reference and user folders
videos = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]
print(f"Found {len(videos)} videos to process")

for video in videos:
    src_path = os.path.join(source_folder, video)
    if video == "proper_crunches.mp4":
        dst_path = os.path.join(reference_folder, video)
        print(f"Moving reference video: {video}")
    else:
        dst_path = os.path.join(user_video_folder, video)
        print(f"Moving user video: {video}")
    
    try:
        shutil.copy(src_path, dst_path)
        print(f"Successfully moved {video}")
    except Exception as e:
        print(f"Error moving {video}: {e}")

# Process reference video
reference_video = os.path.join(reference_folder, "proper_crunches.mp4")
reference_keypoints = os.path.join("keypoints", "reference_proper.npy")

if not os.path.exists(reference_video):
    print("Error: Reference video 'proper_crunches.mp4' not found!")
    exit(1)

print("Processing reference video...")
if not os.path.exists(reference_keypoints):
    try:
        extract_keypoints(reference_video, reference_keypoints)
        print("Reference keypoints extracted successfully")
    except Exception as e:
        print(f"Error extracting reference keypoints: {e}")
        exit(1)

# Process user videos
user_videos = glob.glob(os.path.join(user_video_folder, "*.mp4"))
print(f"Found {len(user_videos)} user videos to analyze")

for user_video in user_videos:
    user_filename = os.path.basename(user_video)
    user_name = os.path.splitext(user_filename)[0]
    
    
    user_key_path = os.path.join("keypoints", f"{user_name}.npy")
    
    try:
        extract_keypoints(user_video, user_key_path)
        errors = compare_keypoints(reference_keypoints, user_key_path)
        summary = summarize_mistakes(errors, total_frames=len(errors), min_frame_threshold=5)
        
        if summary:
            for line in summary:
                print(f"  - {line}")
            
            # Get LLM feedback
            print("\nGenerating AI feedback...")
            feedback = get_llm_feedback("proper_crunches", summary)
            print("\nAI Workout Feedback:")
            print("-" * 30)
            print(feedback)
        else:
            print("No significant mistakes detected!")
            
    except Exception as e:
        print(f"Error processing {user_name}:")
        traceback.print_exc()

print(f"\n{'='*50}")
print("Analysis complete!")
print(f"{'='*50}")