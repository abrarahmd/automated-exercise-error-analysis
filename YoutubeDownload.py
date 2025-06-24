import os
import yt_dlp

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=AIPYd3hfzwg"
]
SAVE_DIR = "clips"

os.makedirs(SAVE_DIR, exist_ok=True)

ydl_opts = {
    'format': 'bestvideo',
    'outtmpl': os.path.join(SAVE_DIR, '%(title).50s.%(ext)s'),
    'quiet': False,
    'noplaylist': True,
    'ignoreerrors': True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in VIDEO_URLS:
        try:
            print(f"\nDownloading: {url}")
            ydl.download([url])
        except Exception as e:
            print(f"Failed to download {url}: {e}")
