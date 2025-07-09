import openai
import base64
from io import BytesIO
from PIL import Image
import os
import glob
import math

def analyse_workout_feedback_sequential(
    reference_dir,
    user_dirs,
    exercise_name,
    output_dir="llm_feedback_output",
    model_name="qwen2.5vl:32b",
    api_base="http://172.16.7.83:11434/v1/",
    api_key="ollama"
):
    openai.base_url = api_base
    openai.api_key = api_key
    os.makedirs(output_dir, exist_ok=True)

    def encode_image(image_path):
        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_segments(frames, segment_count=3):
        total = len(frames)
        seg_len = math.ceil(total / segment_count)
        return [frames[i*seg_len:(i+1)*seg_len] for i in range(segment_count)]

    def select_three_from_segment(segment):
        if len(segment) < 3:
            return segment
        step = len(segment) // 2
        return [segment[0], segment[step], segment[-1]]

    def generate_segment_prompt(exercise_name, segment_number):
        return {
            "type": "text",
            "text": (
                f"You are a certified fitness trainer evaluating segment {segment_number} of a **{exercise_name}** workout.\n\n"
                "You will see reference frames (proper form) followed by user frames (actual performance) for this segment.\n\n"
                "Analyze this segment and provide:\n"
                "1. Key errors and form issues you observe in this segment\n"
                "2. Specific observations about posture, control, timing, and alignment\n"
                "3. Note any improvements or correct techniques you see\n\n"
                "Keep your analysis focused on this specific segment. Be detailed and professional."
            )
        }

    def generate_final_prompt(exercise_name, segment_analyses):
        return {
            "type": "text",
            "text": (
                f"You are a certified fitness trainer providing final feedback for a **{exercise_name}** workout.\n\n"
                "You have analyzed 3 segments of the workout. Here are your segment-by-segment observations:\n\n"
                f"SEGMENT 1 ANALYSIS:\n{segment_analyses[0]}\n\n"
                f"SEGMENT 2 ANALYSIS:\n{segment_analyses[1]}\n\n"
                f"SEGMENT 3 ANALYSIS:\n{segment_analyses[2]}\n\n"
                "Based on all segments, provide exactly **two paragraphs**, each exactly 200 words:\n\n"
                "**Errors:** Comprehensive summary of all errors, mistakes, and form issues observed across all segments. Include posture problems, control issues, timing errors, and alignment mistakes.\n\n"
                "**Suggestions:** Actionable instructions and corrections to help the user improve their form. Provide specific, clear, and professional guidance for better performance.\n\n"
                "Make sure each paragraph is exactly 200 words and covers the complete workout analysis."
            )
        }

    def run_segment_analysis(reference_b64s, user_b64s, exercise_name, segment_number):
        content = [generate_segment_prompt(exercise_name, segment_number)]

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

    def run_final_analysis(exercise_name, segment_analyses):
        content = [generate_final_prompt(exercise_name, segment_analyses)]

        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content

    reference_frames = sorted(glob.glob(os.path.join(reference_dir, "*.jpg")))
    ref_segments = get_segments(reference_frames, segment_count=3)
    ref_b64_segments = [
        [encode_image(f) for f in select_three_from_segment(seg)]
        for seg in ref_segments
    ]

    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        user_frames = sorted(glob.glob(os.path.join(user_dir, "*.jpg")))
        user_segments = get_segments(user_frames, segment_count=3)
        user_b64_segments = [
            [encode_image(f) for f in select_three_from_segment(seg)]
            for seg in user_segments
        ]
        segment_analyses = []
        print(f"Starting sequential analysis for {user_name}...")
        
        for seg_idx in range(3):
            segment_feedback = run_segment_analysis(
                ref_b64_segments[seg_idx],
                user_b64_segments[seg_idx],
                exercise_name=exercise_name,
                segment_number=seg_idx + 1
            )
            segment_analyses.append(segment_feedback)

        print(f"Generating final comprehensive feedback...")
        final_feedback = run_final_analysis(exercise_name, segment_analyses)
        complete_output = (
            "## Final Comprehensive Feedback:\n\n"
            f"{final_feedback}"
        )

        with open(os.path.join(output_dir, f"{user_name}_final_feedback.txt"), "w") as f:
            f.write(complete_output)