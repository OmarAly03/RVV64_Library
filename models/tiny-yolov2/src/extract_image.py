import cv2
import numpy as np
import sys
import os  # Add this for file path manipulation

if len(sys.argv) != 2:
    print("Usage: python3 extract_image.py <IMAGE_PATH>")
    exit()
    
IMAGE_PATH = sys.argv[1]

# Automatically generate the output path based on the input file name
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]  # Extract base name without extension
OUTPUT_PATH = f"image_binaries/{base_name}.bin"

NET_H, NET_W = 416, 416

print(f"Loading and processing {IMAGE_PATH}...")

# 1. Load original image
original_image = cv2.imread(IMAGE_PATH)
if original_image is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit()

# 2. Resize to the network input size
image_resized = cv2.resize(original_image, (NET_W, NET_H))

# 3. Convert from BGR to RGB
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 4. Convert to float32
# (Remember: no / 255.0 because our model has the scaler built-in)
image_float = image_rgb.astype(np.float32)

# 5. Transpose from (H, W, C) to (C, H, W)
#    OpenCV blobFromImage can do this, but we'll do it manually
#    to match the (N, C, H, W) format.
image_chw = np.transpose(image_float, (2, 0, 1))

# 6. Add a batch dimension (N=1)
#    Final shape is (1, 3, 416, 416)
input_tensor = np.expand_dims(image_chw, axis=0)

# 7. Save as a flat binary file
print(f"Saving input tensor with shape {input_tensor.shape} to {OUTPUT_PATH}...")
input_tensor.tofile(OUTPUT_PATH)

print("Done.")