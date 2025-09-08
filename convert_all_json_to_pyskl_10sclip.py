import json
import pickle
import numpy as np
import os
import random

# Configurations

JSON_FOLDER = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Stick Figures RDG\WallS\Final procedures\Poses RDG"
OUTPUT_FOLDER = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Stick Figures RDG\WallS\Final procedures\Pyskl_data\dataset_10seconds"


IMG_SHAPE = (1080, 1920)  # Image dimensions
LABEL = 1  # Placeholder action label
N = 5  # Sampling every Nth frame
SEGMENT_LENGTH = 250  # 10s segment (250 frames at 25 fps)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all JSON files
json_files = sorted([os.path.join(JSON_FOLDER, f) for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])
#json_files=[json for json in json_files if json.endswith(r"0cccdf49-746d-54a7-9cc7-dcc13de4c259.json")]
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {JSON_FOLDER}")

# Shuffle for randomness
random.shuffle(json_files)

# Split into xsub and xview subsets randomly (50% each)
mid_idx = len(json_files) // 2
xsub_files = json_files
xview_files = json_files[mid_idx:]

def split_train_val(file_list):
    split_idx = int(len(file_list) * 0.2)
    return file_list[:split_idx], file_list

xsub_train, xsub_val = split_train_val(xsub_files)
xview_train, xview_val = split_train_val(xview_files)

# Function to clean keypoints
def clean_keypoints(keypoints):
    for v in range(len(keypoints) // 3):  # [x, y, confidence]
        x, y, confidence = keypoints[v * 3:v * 3 + 3]
        keypoints[v * 3] = 0 if np.isinf(x) or np.isnan(x) else x
        keypoints[v * 3 + 1] = 0 if np.isinf(y) or np.isnan(y) else y
        keypoints[v * 3 + 2] = 0 if np.isinf(confidence) or np.isnan(confidence) else confidence
    return keypoints
j=1
# Process each JSON file separately
for json_file in json_files:
    frame_dir = os.path.splitext(os.path.basename(json_file))[0]
    output_filename = f"{frame_dir}.pkl"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Skip processing if output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {frame_dir} - already processed.")
        continue

    # Determine the split category
    if json_file in xsub_val:
        SPLIT_NAME = "xsub_val"
    elif json_file in xsub_train:
        SPLIT_NAME = "xsub_train"
    elif json_file in xview_train:
        SPLIT_NAME = "xview_train"
    else:
        SPLIT_NAME = "xview_val"

    # Load JSON file
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Processing {frame_dir}..., {j} out of {len(json_files)}")
            j+=1
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading {json_file}: {e}")
        continue  # Skip to the next file

    frames = sorted(map(int, data.keys()))
    sampled_frames = frames  # Keep all frames
    num_segments = max(1, len(sampled_frames) // SEGMENT_LENGTH)  # Ensure at least one segment
    all_annotations = []
    split = {"xsub_train": [], "xsub_val": [], "xview_train": [], "xview_val": []}
    
    for i, seg_idx in enumerate(range(num_segments)):
        segment_frames = sampled_frames[seg_idx * SEGMENT_LENGTH : (seg_idx + 1) * SEGMENT_LENGTH]
        person_ids = sorted({p["id"] for frame in segment_frames for person in data.get(str(frame), {}).values() for p in person})
        M, T, V, C = max(1, len(person_ids)), len(segment_frames), 17, 2  # Ensure at least one person
        keypoint_array = np.zeros((M, T, V, C), dtype=np.float16)
        keypoint_score_array = np.zeros((M, T, V), dtype=np.float16)
        person_id_map = {pid: i for i, pid in enumerate(person_ids)}

        has_keypoints = False  # Track if we have valid keypoints
        for t, frame in enumerate(segment_frames):
            for person in data.get(str(frame), {}).values():
                for p in person:
                    pid = p["id"]
                    if pid in person_id_map:
                        m = person_id_map[pid]
                        keypoints = clean_keypoints(p["keypoints"])
                        for v in range(V):
                            keypoint_array[m, t, v] = keypoints[v * 3:v * 3 + 2]
                            keypoint_score_array[m, t, v] = keypoints[v * 3 + 2]
                        has_keypoints = True  # Mark that we have at least one keypoint
        
        # If no keypoints exist, add a dummy person with zero keypoints
        if not has_keypoints:
            print(f"Warning: No keypoints found for segment {frame_dir}{i}, adding dummy keypoints.")
            keypoint_array = np.zeros((1, T, V, C), dtype=np.float16)  # 1 dummy person
            keypoint_score_array = np.zeros((1, T, V), dtype=np.float16)

        # Assign to the correct split
        split[SPLIT_NAME].append(frame_dir+str(i))

        # Store annotation for this segment
        all_annotations.append({
            "frame_dir": frame_dir+str(i),
            "total_frames": T,
            "img_shape": IMG_SHAPE,
            "original_shape": IMG_SHAPE,
            "label": LABEL,
            "keypoint": keypoint_array,
            "keypoint_score": keypoint_score_array,
        })

    # Save one pkl file for the entire JSON file
   # output_filename = f"{frame_dir}.pkl"
   # output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    output_data = {
        "split": split,
        "annotations": all_annotations
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved {output_filename} to {OUTPUT_FOLDER}")

print("Processing complete.")
