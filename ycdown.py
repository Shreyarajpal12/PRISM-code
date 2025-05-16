import os
import subprocess
from datasets import load_dataset

# 1. Create 'videos_youcook2' folder if it doesn't exist
os.makedirs("videos_youcook2", exist_ok=True)

# 2. Load the YouCook2 validation split
dataset = load_dataset("lmms-lab/YouCook2", split="val")

# 3. Track already downloaded video IDs (from local folder only)
downloaded = set(os.path.splitext(f)[0] for f in os.listdir("videos_youcook2") if f.endswith(".mp4"))

# 4. Loop over dataset and download only new videos
for item in dataset:
    url = item["video_url"]
    video_id = url.split("v=")[-1]
    filename = f"{video_id}.mp4"
    filepath = os.path.join("videos_youcook2", filename)

    if video_id not in downloaded:
        print(f"Downloading {video_id}...")
        try:
            subprocess.run(["yt-dlp", url, "-o", filepath], check=True)
            downloaded.add(video_id)
        except subprocess.CalledProcessError:
            print(f"❌ Failed to download {video_id}")
    else:
        print(f"✅ Already downloaded: {video_id}")
