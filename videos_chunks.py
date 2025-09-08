import os
import subprocess
from pathlib import Path

# Set the full paths to ffmpeg and ffprobe executables
ffmpeg_path = r"C:\Users\emanuelefrassi\Downloads\ffmpeg-2025-06-11-git-f019dd69f0-full_build\bin\ffmpeg.exe"
ffprobe_path = r"C:\Users\emanuelefrassi\Downloads\ffmpeg-2025-06-11-git-f019dd69f0-full_build\bin\ffprobe.exe"

# Define input and output directories
input_dir = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos")
output_dir = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s_2")

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# List all .mkv video files in the input directory
video_files = list(input_dir.glob("*.mkv"))
num=0
# Loop over each video
for video_path in video_files:
    num+=1
    video_name = video_path.stem  # name without extension
    subfolder = output_dir / video_name
    subfolder.mkdir(exist_ok=True)

    # Get video duration using ffprobe
    cmd_duration = [
        ffprobe_path, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd_duration, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    
    # Number of 10-second chunks
    num_chunks = int(duration // 10)

    for i in range(num_chunks):
        start_time = i * 10
        output_filename = subfolder / f"{video_name}{i:03}.mkv"
        
        cmd_cut = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time), "-i", str(video_path),
            "-t", "10", "-c", "copy", str(output_filename)
        ]
        subprocess.run(cmd_cut)
    print(f"✅ Done splitting video {num} out of {len(video_files)}")

print("✅ Done splitting all videos into 10-second chunks.")
