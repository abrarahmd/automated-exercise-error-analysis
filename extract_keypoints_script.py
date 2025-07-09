import numpy as np
import cv2
import mediapipe as mp
import os

SELECTED_JOINTS = [
    0, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30
]

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