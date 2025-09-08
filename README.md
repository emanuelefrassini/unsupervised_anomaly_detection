# Unsupervised Anomaly Detection
This repository contains my personal scripts and code for my anomaly detection paper. The pyskl repository (used as feature extraction module) is included here as a Git submodule.

## Folder Structure

• pyskl/: The pyskl library, included as a Git submodule.

• requirements.txt: A list of all required Python packages.

• concatenate_pose_rgb.py: Script to concatenate pose and RGB data.

• convert_all_json_to_pyskl_10sclip.py: Converts JSON data to the pyskl 10-second clip format.

• create_conf_files.py: Creates configuration files for the project.

• create_conf_files_videos.py: Creates video-specific configuration files.

• process_videos_optimized_2.py: Optimized script for video processing.

• test.py: The feature extraction script.

• VAE_reproducible_improved.py: The main script that detects anomalies with features extracted from 10s clip.

• videos_chunks.py: Script for splitting videos in chunks.

## Setup Instructions

### Prerequisites

Before setting up the project, you need to install ffprobe and ffmpeg. These are command-line tools used for video processing.

macOS (using Homebrew):
```
brew install ffmpeg
```
Ubuntu/Debian:
```
sudo apt update
sudo apt install ffmpeg
```
Windows (using Chocolatey):
```
choco install ffmpeg
```
### Clone the Repository

Clone this repository and its submodules using the following command:
```
git clone --recurse-submodules https://www.google.com/search?q=https://github.com/emanuelefrassini/unsupervised_anomaly_detection.git
cd anomalydetection
```
### Install Dependencies

Install all the required Python packages from the requirements.txt file.
```
pip install -r requirements.txt
```
