import cv2
import numpy as np
import sys
import os
import glob

if len(sys.argv) != 3:
    print("Usage: python3 extract_image.py <INPUT_FOLDER> <OUTPUT_FOLDER>")
    print("Example: python3 extract_image.py /path/to/images ../image_binaries")
    exit()

INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]

NET_H, NET_W = 416, 416

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported image extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

# Find all image files in the input folder
image_files = []
for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, extension)))
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, extension.upper())))

if not image_files:
    print(f"No image files found in {INPUT_FOLDER}")
    print(f"Supported formats: {', '.join([ext.replace('*', '') for ext in image_extensions])}")
    exit()

print(f"Found {len(image_files)} image(s) in {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")

processed_count = 0
failed_count = 0

for image_path in image_files:
    try:
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.bin")
        
        print(f"Processing: {os.path.basename(image_path)}...")
        
        # 1. Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  ERROR: Could not load {image_path}")
            failed_count += 1
            continue
        
        # 2. Resize to the network input size
        image_resized = cv2.resize(original_image, (NET_W, NET_H))
        
        # 3. Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # 4. Convert to float32
        image_float = image_rgb.astype(np.float32)
        
        # 5. Transpose from (H, W, C) to (C, H, W)
        image_chw = np.transpose(image_float, (2, 0, 1))
        
        # 6. Add a batch dimension (N=1)
        #    Final shape is (1, 3, 416, 416)
        input_tensor = np.expand_dims(image_chw, axis=0)
        
        # 7. Save as a flat binary file
        input_tensor.tofile(output_path)
        
        print(f"  -> Saved: {os.path.basename(output_path)} (shape: {input_tensor.shape})")
        processed_count += 1
        
    except Exception as e:
        print(f"  ERROR processing {image_path}: {str(e)}")
        failed_count += 1

print(f"\nProcessing complete!")
print(f"Successfully processed: {processed_count} images")
print(f"Failed: {failed_count} images")
print(f"Output files saved in: {OUTPUT_FOLDER}")