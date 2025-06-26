import numpy as np

def compare_keypoints(reference_path, user_path, threshold=0.1):
    ref = np.load(reference_path)
    usr = np.load(user_path)

    min_len = min(len(ref), len(usr))
    ref = ref[:min_len]
    usr = usr[:min_len]

    joint_errors = []

    for f in range(min_len):
        frame_errors = []
        for j in range(33):
            rx, ry, rz, _ = ref[f][j]
            ux, uy, uz, _ = usr[f][j]
            dist = np.sqrt((rx - ux)**2 + (ry - uy)**2 + (rz - uz)**2)
            if dist > threshold:
                frame_errors.append((j, dist))
        joint_errors.append(frame_errors)

    return joint_errors
