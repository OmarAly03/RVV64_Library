from PIL import Image
import numpy as np

# --- 1. Define Image Path and Output Path ---
# IMPORTANT: Change this to the path of your PNG file
image_path = '../images/0.png' 
output_path = '../image_binaries/0.bin'

# --- 2. Define Preprocessing Constants ---
MEAN = 0.1307
STD_DEV = 0.3081
TARGET_H, TARGET_W = 32, 32

try:
    # --- 3. Load and Process the Image ---
    print(f"Loading image: {image_path}...")
    
    # Open the image using PIL
    img = Image.open(image_path)

    # 1. Ensure it's grayscale (L = Luminance)
    # This replaces transforms.Grayscale()
    img = img.convert('L')
    
    # 2. Resize to 32x32
    # This replaces transforms.Resize()
    # Image.Resampling.LANCZOS is a high-quality filter, good for this.
    img = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

    # 3. Convert to numpy array and scale to [0, 1]
    # This replaces transforms.ToTensor()
    # We set the type to float32 right away
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 4. Normalize
    # This replaces transforms.Normalize()
    img_normalized = (img_np - MEAN) / STD_DEV

    # 5. Add Batch and Channel dimensions
    # This replaces .unsqueeze(0)
    # Shape goes from (32, 32) to (1, 1, 32, 32)
    final_array = img_normalized.reshape(1, 1, TARGET_H, TARGET_W)

    print(f"Image processed. Final array shape: {final_array.shape}")

    # --- 6. Save to .bin ---
    # The array is already np.float32, so we just save it
    final_array.tofile(output_path)

    print(f"\nSuccessfully saved preprocessed image to: {output_path}")
    print("This file is now ready to be loaded by your C++ program.")

except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'.")
    print("Please make sure the file is in the correct location.")
except Exception as e:
    print(f"An error occurred: {e}")