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
├──
├──
├── ffmpeg/
├── main.py # Main analysis script
├── extract_pose.py
├── compare.py
├── feedback.py
└── README.md
```
---
## 🚀 Pipeline Execution

```bash
pip install numpy==1.24.4
pip install opencv-python easyocr
```
create .env file and store your groq api key there.
```bash
pip install mediapipe
pip install python-dotenv
```

