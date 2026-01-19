import onnx
import onnx.numpy_helper
import numpy as np

# Load the ONNX model
model = onnx.load("lenet.onnx")

# Get the initializers (these are the trained parameters)
params = model.graph.initializer

print(f"Found {len(params)} parameter tensors. Extracting...")

# Iterate through each parameter and save it
for param in params:
    # Get the weights as a NumPy array
    weights_array = onnx.numpy_helper.to_array(param)
    
    # Get the name and shape
    param_name = param.name
    param_shape = param.dims
    
    # Define a clean filename (e.g., "fc1.bias.bin")
    filename = f"{param_name}.bin"
    
    print(f"  -> Saving {filename} (Shape: {param_shape})")
    
    # Save the weights as a raw binary file of float32
    weights_array.astype(np.float32).tofile(filename)

print("\nDone. All parameters have been saved as .bin files.")