import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
input_root = 'E:/Data/Metalens/Journal/lr_metalens_3mm'         # Folder with original images
output_root = 'E:/Data/Metalens/Journal/lr_metalens_3mm_corrected'    # Where corrected images will be saved
correction_map = np.load('E:/Data/Metalens/Journal/lr_metalens_3mm/correction_matrix.npy')  # Load your 2D correction map (shape: H x W)
correction_map = np.transpose(correction_map, (1, 2, 0)) * 0.95

# --- Supported image extensions ---
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# --- Create output root if needed ---
os.makedirs(output_root, exist_ok=True)

# --- Walk through all images ---
image_paths = [os.path.join(root, f)
            for root, _, files in os.walk(input_root)
            for f in files if os.path.splitext(f)[-1].lower() in valid_exts]

for input_path in tqdm(image_paths):
    # Construct output path
    rel_path = os.path.relpath(input_path, input_root)
    output_path = os.path.join(output_root, rel_path)

    # Ensure output subfolder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0

    # Resize correction map if needed (should match image shape)

    # Apply correction (broadcast across channels)
    corrected = np.clip(img * correction_map, 0, 1)

    # Save corrected image
    cv2.imwrite(output_path, (corrected * 255).astype(np.uint8))

    # print(f"Saved: {output_path}")
