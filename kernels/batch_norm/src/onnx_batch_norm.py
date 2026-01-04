import os
import onnx
from onnx import helper, TensorProto
import numpy as np
import sys

# Get channel count from command line arguments or use default
num_channels = 3  # Default
if len(sys.argv) >= 3:
    num_channels = int(sys.argv[2])  # C parameter

print(f"Creating BatchNorm ONNX model for {num_channels} channels")

# BatchNorm typically works on 4D tensors (NCHW format)
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, num_channels, None, None])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, num_channels, None, None])

# Create constant tensors for BatchNorm parameters
# Scale (gamma) - ones
scale_values = np.ones(num_channels, dtype=np.float32)
scale_tensor = helper.make_tensor(
    name="scale",
    data_type=TensorProto.FLOAT,
    dims=[num_channels],
    vals=scale_values.flatten().tolist()
)

# Bias (beta) - zeros  
bias_values = np.zeros(num_channels, dtype=np.float32)
bias_tensor = helper.make_tensor(
    name="bias",
    data_type=TensorProto.FLOAT,
    dims=[num_channels],
    vals=bias_values.flatten().tolist()
)

# Running mean - zeros
mean_values = np.zeros(num_channels, dtype=np.float32)
mean_tensor = helper.make_tensor(
    name="mean",
    data_type=TensorProto.FLOAT,
    dims=[num_channels],
    vals=mean_values.flatten().tolist()
)

# Running variance - ones
var_values = np.ones(num_channels, dtype=np.float32)
var_tensor = helper.make_tensor(
    name="var",
    data_type=TensorProto.FLOAT,
    dims=[num_channels],
    vals=var_values.flatten().tolist()
)

batch_norm_node = helper.make_node(
    "BatchNormalization",
    inputs=["input", "scale", "bias", "mean", "var"],
    outputs=["output"],
    epsilon=1e-05,  # small constant to avoid division by zero
    momentum=0.9    # momentum for running statistics
)

graph = helper.make_graph(
    nodes=[batch_norm_node],
    name="BatchNormalizationGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[scale_tensor, bias_tensor, mean_tensor, var_tensor]
)

model = helper.make_model(graph, producer_name="batch_norm_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "batch_norm.onnx")

onnx.save(model, output_path)
print(f"BatchNormalization ONNX model saved to: {output_path}")