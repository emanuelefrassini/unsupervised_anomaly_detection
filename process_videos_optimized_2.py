
import subprocess
from pathlib import Path
import pandas as pd

input_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s\temp_input")
output_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s_processed\temp_output")

def log_message(msg):
    with open(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose\debug_log.txt", "a") as f:
        f.write(msg + "\n")

def can_open_video(path):
    log_message(f"Trying to open {path}")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_shape(input_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(input_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    w, h = map(int, output.split('x'))
    return w, h

def compress_video(input_path_str):
    input_path = Path(input_path_str)
    relative_path = input_path.relative_to(input_root)
    output_path = output_root / relative_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_message(f"Output path is {output_path}")

    # If already exists and is readable, skip
    if output_path.exists() and can_open_video(output_path):
        print(f"Video opens fine, skipping: {input_path}")
        log_message(f"Skipping already valid: {output_path}")
        return None

    try:
        w, h = get_shape(input_path)
        target_size = 540
        scale_str = f"-vf scale=-2:{target_size}" if w >= h else f"-vf scale={target_size}:-2"

        cmd_cpu = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            *scale_str.split(),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            str(output_path)
        ]
        subprocess.run(cmd_cpu, check=True)
        print(f"Compressed video: {input_path} → {output_path}")
        log_message(f"Compressed: {input_path} to {output_path}")
        return None

    except subprocess.CalledProcessError as e_cpu:
        msg = f"Failed to compress {input_path} with CPU encoding: {e_cpu}"
        print(msg)
        log_message(msg)
        return str(input_path)
    except Exception as e:
        msg = f"Failed to process {input_path}: {e}"
        print(msg)
        log_message(msg)
        return str(input_path)

# Read failed folders from text file
file_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose\failed_files.txt"
with open(file_path, 'r') as f:
    lines = f.readlines()
failed_fileslist = [line.strip().split(":")[1].strip() for line in lines if "Failed config:" in line]

# Main processing
if __name__ == "__main__":
    video_paths = [p for p in input_root.rglob("*.mkv")]# if p.parent.name in failed_fileslist]
    failed_files = []

    for path in video_paths:
        result = compress_video(str(path))
        if result is not None:
            failed_files.append(result)

    failed_folders = set(Path(f).parent.name for f in failed_files)

    if failed_folders:
        df_failed = pd.DataFrame({"Failed folders": list(failed_folders)})
        df_failed.to_excel("failed_folders.xlsx", index=False)
        print("Saved failed folders to failed_folders.xlsx")
    else:
        print("No failed files detected.")

    input_folders = {p.name for p in input_root.iterdir() if p.is_dir()}
    output_folders = {p.name for p in output_root.iterdir() if p.is_dir()}

    missing_folders = input_folders - output_folders - failed_folders

    if missing_folders:
        print(f"Missing folders (not processed and not failed): {missing_folders}")
        df_missing = pd.DataFrame({"Missing folders": list(missing_folders)})
        df_missing.to_excel("missing_folders.xlsx", index=False)
    else:
        print("All folders accounted for (either processed or failed).")



r'''
import subprocess
from pathlib import Path
import multiprocessing as mp
import pandas as pd

input_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s\816\816153")
output_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s_processed\816\816153")

def get_shape(input_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(input_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    w, h = map(int, output.split('x'))
    return w, h

def compress_video(input_path_str):
    input_path = Path(input_path_str)
    relative_path = input_path.relative_to(input_root)
    output_path = output_root / relative_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
   # output_path = output_path.name.replace("_part", "")

    if output_path.exists():
     #   print(f"Skipping existing: {output_path}")
        return None

    try:
        w, h = get_shape(input_path)
        target_size = 540
        scale_str = f"-vf scale=-2:{target_size}" if w >= h else f"-vf scale={target_size}:-2"

        # CPU encoding only
        cmd_cpu = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            *scale_str.split(),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            str(output_path)
        ]
        subprocess.run(cmd_cpu, check=True)
        print(f"Compressed with CPU: {input_path} → {output_path}")
        return None

    except subprocess.CalledProcessError as e_cpu:
        print(f"Failed to process {input_path} with CPU encoding: {e_cpu}")
        return str(input_path)
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return str(input_path)

file_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose\failed_files.txt"

# Read and extract lines
with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract only the lines that contain "Failed config:" and get the file number
failed_fileslist = [line.strip().split(":")[1].strip() for line in lines if "Failed config:" in line]

if __name__ == "__main__":
 #   video_paths = list(input_root.rglob("*.mkv"))
    video_paths = [p for p in input_root.rglob("*.mkv") if p.parent.name in failed_fileslist]

    with mp.Pool(processes=4) as pool:
        failed_files = pool.map(compress_video, [str(p) for p in video_paths])

    failed_files = [f for f in failed_files if f is not None]
    failed_folders = set(Path(f).parent.name for f in failed_files)

    if failed_folders:
        df_failed = pd.DataFrame({"Failed folders": list(failed_folders)})
        df_failed.to_excel("failed_folders.xlsx", index=False)
        print(f"Saved failed folders to failed_folders.xlsx")
    else:
        print("No failed files detected.")
# Get all input and output folders
    input_folders = {p.name for p in input_root.iterdir() if p.is_dir()}
    output_folders = {p.name for p in output_root.iterdir() if p.is_dir()}
    
    # Folders that were not created at all
    missing_folders = input_folders - output_folders - failed_folders
    
    if missing_folders:
        print(f"Missing folders (not processed and not failed): {missing_folders}")
        df_missing = pd.DataFrame({"Missing folders": list(missing_folders)})
        df_missing.to_excel("missing_folders.xlsx", index=False)
    else:
        print("All folders accounted for (either processed or failed).")


import subprocess
from pathlib import Path
import multiprocessing as mp
import pandas as pd

input_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s")
output_root = Path(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Videos_10s_processed")
def log_message(msg):
    with open(r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose\debug_log.txt", "a") as f:
        f.write(msg + "\n")

def can_open_video(path):
    log_message(f"Trying to open {path}")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
def get_shape(input_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(input_path)
    ]
    output = subprocess.check_output(cmd).decode().strip()
    w, h = map(int, output.split('x'))
    return w, h

def compress_video(input_path_str):
    input_path = Path(input_path_str)
    
    # First try to open video without compressing
 
    
    relative_path = input_path.relative_to(input_root)
    output_path = output_root / relative_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_message(f"output path is {output_path}")
    if output_path.exists() & can_open_video(output_path):
        # Video opens fine - skip compressing
        print(f"Video opens fine, skipping: {input_path}")
        return None
    #if output_path.exists():
     #   print(f"Output already exists, skipping: {output_path}")
      #  return None

    try:
        w, h = get_shape(input_path)
        target_size = 540
        scale_str = f"-vf scale=-2:{target_size}" if w >= h else f"-vf scale={target_size}:-2"

        cmd_cpu = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            *scale_str.split(),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            str(output_path)
        ]
        subprocess.run(cmd_cpu, check=True)
        print(f"Compressed video: {input_path} → {output_path}")
        return None

    except subprocess.CalledProcessError as e_cpu:
        print(f"Failed to compress {input_path} with CPU encoding: {e_cpu}")
        return str(input_path)
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return str(input_path)

file_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose\failed_files.txt"

# Read and extract lines
with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract filenames from "Failed config:" lines
failed_fileslist = [line.strip().split(":")[1].strip() for line in lines if "Failed config:" in line]

if __name__ == "__main__":
    # Filter videos whose filename is in failed_fileslist
    video_paths = [p for p in input_root.rglob("*.mkv") if p.parent.name in failed_fileslist]

    with mp.Pool(processes=4) as pool:
        failed_files = pool.map(compress_video, [str(p) for p in video_paths])

    failed_files = [f for f in failed_files if f is not None]
    failed_folders = set(Path(f).parent.name for f in failed_files)

    if failed_folders:
        df_failed = pd.DataFrame({"Failed folders": list(failed_folders)})
        df_failed.to_excel("failed_folders.xlsx", index=False)
        print(f"Saved failed folders to failed_folders.xlsx")
    else:
        print("No failed files detected.")
    
    input_folders = {p.name for p in input_root.iterdir() if p.is_dir()}
    output_folders = {p.name for p in output_root.iterdir() if p.is_dir()}
    
    missing_folders = input_folders - output_folders - failed_folders
    
    if missing_folders:
        print(f"Missing folders (not processed and not failed): {missing_folders}")
        df_missing = pd.DataFrame({"Missing folders": list(missing_folders)})
        df_missing.to_excel("missing_folders.xlsx", index=False)
    else:
        print("All folders accounted for (either processed or failed).")


#%%
'''