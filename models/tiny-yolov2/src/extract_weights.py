import onnx
import onnx.numpy_helper
import numpy as np
import os

MODEL_PATH = 'onnx_model/tinyyolov2.onnx'
WEIGHTS_DIR = 'model_parameters/'

# Create the weights directory if it doesn't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
model = onnx.load(MODEL_PATH)

manifest_lines = []

print(f"Extracting and saving weights to '{WEIGHTS_DIR}/'...")

# The initializers are the stored parameters (weights, biases, etc.)
for initializer in model.graph.initializer:
    name = initializer.name
    
    # Sanitize the name to be a valid filename
    filename = name.replace('/', '_').replace(':', '_') + '.bin'
    filepath = os.path.join(WEIGHTS_DIR, filename)
    
    # Convert the ONNX tensor to a NumPy array
    try:
        arr = onnx.numpy_helper.to_array(initializer)
    except Exception as e:
        print(f"Error converting {name}: {e}")
        continue
        
    # Save the array as a flat binary file in float32 format
    arr.astype(np.float32).tofile(filepath)
    
    # Get shape info for the manifest
    shape = list(arr.shape)
    shape_str = ",".join(map(str, shape)) if shape else "scalar"
    
    # Add to our manifest
    manifest_line = f"Name: {name}, Filename: {filename}, DType: {arr.dtype}, Shape: [{shape_str}]"
    manifest_lines.append(manifest_line)
    
    print(f"  Saved {name} (Shape: {shape}) to {filename}")

# Save the manifest file
manifest_path = os.path.join(WEIGHTS_DIR, 'weights_manifest.txt')
with open(manifest_path, 'w') as f:
    f.write("\n".join(manifest_lines))

print(f"\nSuccessfully extracted {len(manifest_lines)} parameter tensors.")
print(f"Manifest file saved to: {manifest_path}")