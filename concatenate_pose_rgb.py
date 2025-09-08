import os
import pickle
from tqdm import tqdm
import numpy as np

# Paths
rgbpose_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose_rgb50_pose200_rgb_NORMAL"
slowonly_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose_rgb50_pose200_pose_NORMAL"
output_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\pose_rgb_2_NORMAL"

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file
for filename in tqdm(os.listdir(rgbpose_dir)):
    if filename.endswith(".pkl"):
        try:
            rgbpose_path = os.path.join(rgbpose_dir, filename)
            slowonly_path = os.path.join(slowonly_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load both files
            with open(rgbpose_path, 'rb') as f:
                rgbpose_list = pickle.load(f)

            with open(slowonly_path, 'rb') as f:
                slowonly_list = pickle.load(f)

            # Check length consistency
            if len(rgbpose_list) != len(slowonly_list):
                print(f"poses are {len(slowonly_list)} and videos are {len(rgbpose_list)}")
               # slowonly_list=slowonly_list[:len(rgbpose_list)]
                #raise ValueError(f"Length mismatch in {filename}")

            # Normalize and concatenate
            combined = [
                np.concatenate((
                    normalize_vector(np.array(rgb)),
                    normalize_vector(np.array(slow))
                ), axis=0)
                for rgb, slow in zip(rgbpose_list, slowonly_list)
            ]

            # Save the combined list
            with open(output_path, 'wb') as f:
                pickle.dump(combined, f)

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
