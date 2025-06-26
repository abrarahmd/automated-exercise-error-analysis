# 🏋️‍♂️ Automated Exercise Error Analysis

A simple end-to-end Python pipeline to:

- ✅ Download workout videos from YouTube  
- ✂️ Trim clips  
- ❌ Remove text overlays  
- 📊 Analyze exercise mistakes using pose detection  

---

## 🔧 Installation & Setup

### 1. 🐍 Python Environment

1. Create a folder named `python/` in your project directory.  
2. Install **Python 3.9** in that folder (use [official Python site](https://www.python.org/downloads/release/python-390/)).  
3. Activate your virtual environment.

Then install required packages:

```bash
pip install yt_dlp
pip install ffmpeg-python
```
### 2. 🧰 Install FFmpeg
1. Go to https://www.gyan.dev/ffmpeg/builds/
2. Download ffmpeg-release-essentials.zip 
3. Extract it and rename the folder to ffmpeg
4. Add the ffmpeg/bin directory to your system PATH.
---
## 📁 Folder Structure
```
Automated Exercise Error Analysis/
│
├── python/ 
├── ffmpeg/
├── yt_clips_text_removed/ # Processed user clips (text removed)
├── yt_download.py # Script to download YouTube videos
├── yt_trim.py # Script to trim downloaded videos
├── rm_txt.py # Script to remove text from clips
├── main.py # Main analysis script
├── extract_pose.py
├── compare.py
├── feedback.py
└── README.md
```
---
## 🚀 Pipeline Execution
```bash
python yt_download.py
```
Then rename the downloaded video to ytv_downloaded.mp4
```bash
python yt_trim.py
```
Then rename the trimmed clips according to the exercise types/mistakes.
```bash
pip install numpy==1.24.4
pip install opencv-python easyocr
python rm_txt.py
```
```bash
pip install mediapipe
python main.py
```
