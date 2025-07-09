# 🏋️‍♂️ Automated Exercise Error Analysis

A simple end-to-end Python pipeline to analyze exercise mistakes using keypoints and VLM

---

## 🔧 Installation & Setup

### 1. 🐍 Python Environment

1. Create a folder named `python/` in your project directory.  
2. Install **Python 3.9** in that folder (use [official Python site](https://www.python.org/downloads/release/python-390/)).  
3. Activate your virtual environment.
```bash
python/python -m venv venv
venv\Scripts\Activate.ps1          
```
Then install required packages:

```bash
pip install ffmpeg-python
```
### 2. 🧰 Install FFmpeg
1. Go to https://www.gyan.dev/ffmpeg/builds/
2. Download ffmpeg-release-essentials.zip 
3. Extract it and add the ffmpeg/bin directory to your system PATH.
---
## 📁 Folder Structure
```
Automated Exercise Error Analysis/
│
├── python/
├── clips
├── ffmpeg-7.1.1-essentials_build/
├── analyse_workout.py
├── extract_frames_from_videos_script.py
├── extract_keypoints_script.py
├── keypoints_to_videos_script.py
├── main.py # Main Script
└── README.md

```
---
## 🚀 Pipeline Execution

```bash
pip install numpy==1.24.4
pip install opencv-python easyocr
pip install mediapipe
pip install python-dotenv
python main.py
```

## 🎯 Results

After executing the `main.py` script, the following steps are performed automatically:

✅ **Video Preprocessing**  
- All input videos are converted to **5 FPS** and **480p resolution**.  
- Reference videos are stored in the `reference/` folder.  
- User videos are stored in the `user_videos/` folder.  

✅ **Keypoint Extraction**  
- Keypoints are extracted for each video and saved as `.npy` files in the `keypoints/` folder.  

✅ **Video Generation (Annotated)**  
- Videos (with keypoints) are generated from the extracted keypoints and saved in the `final_videos/` folder.  

✅ **Frame Extraction**  
- Frames are extracted from each video with keypoints and saved as images in the `final_frames/` folder.  

✅ **Batch Comparison**  
- Both reference frames and user frames are divided into **3 batches**.  
- For each batch, the first 3 reference frames are compared with the first 3 user frames in that batch.  

✅ **VLM Analysis**  
- For each batch, the Vision-Language Model (VLM) analyzes the comparisons, identifies mistakes, and generates error descriptions and correction suggestions.  
- After all batches are processed, the VLM summarizes all errors and suggestions into:  
  - A **final error analysis paragraph**  
  - A list of **final suggestions for improvement**  

✅ **Output**  
- The final feedback is saved in the `workout_feedback/` folder.  
