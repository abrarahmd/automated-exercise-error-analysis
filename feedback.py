import numpy as np
import cv2
import mediapipe as mp
import os

# List of joint indices you want to use
SELECTED_JOINTS = [
    0, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30
]

# Mapping from joint index to readable name
JOINT_NAMES = {
    0: "nose",
    7: "left ear",
    8: "right ear",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel"
}

def extract_keypoints(video_path, save_path):
    mp_pose = mp.solutions.pose
    keypoints_all = []

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = [(landmarks[i].x, landmarks[i].y, landmarks[i].z, landmarks[i].visibility) for i in SELECTED_JOINTS]
            else:
                keypoints = [(0, 0, 0, 0)] * len(SELECTED_JOINTS)

            keypoints_all.append(keypoints)

    cap.release()
    np.save(save_path, np.array(keypoints_all))

def compare_keypoints(reference_path, user_path, threshold=0.1):
    ref = np.load(reference_path)
    usr = np.load(user_path)

    min_len = min(len(ref), len(usr))
    ref = ref[:min_len]
    usr = usr[:min_len]

    joint_errors = []

    for f in range(min_len):
        frame_errors = []
        for j in range(len(SELECTED_JOINTS)):
            rx, ry, rz, _ = ref[f][j]
            ux, uy, uz, _ = usr[f][j]
            dist = np.sqrt((rx - ux)**2 + (ry - uy)**2 + (rz - uz)**2)
            if dist > threshold:
                joint_id = SELECTED_JOINTS[j]
                frame_errors.append((joint_id, dist))
        joint_errors.append(frame_errors)

    return joint_errors

def summarize_mistakes(errors, total_frames=None, min_frame_threshold=5):
    joint_counts = {}
    for frame_errors in errors:
        for joint_id, _ in frame_errors:
            joint_counts[joint_id] = joint_counts.get(joint_id, 0) + 1

    result = []
    for joint_id, count in sorted(joint_counts.items(), key=lambda x: -x[1]):
        if count >= min_frame_threshold:
            joint_name = JOINT_NAMES.get(joint_id, f"joint {joint_id}")
            if total_frames:
                percentage = (count / total_frames) * 100
                result.append(f"- {joint_name} misaligned in {count} frames ({percentage:.1f}%)")
            else:
                result.append(f"- {joint_name} misaligned in {count} frames")

    if not result:
        result.append("No significant posture mistakes detected.")

    return result
