import os
import onnx
from onnx import helper, TensorProto
import sys

# Get parameters from command line arguments or use defaults
kernel_h, kernel_w = 2, 2  # Default kernel size
stride_h, stride_w = 1, 1  # Default stride
pad_h, pad_w = 0, 0       # Default padding

if len(sys.argv) >= 7:  # Expecting: script N C H W kH kW [sH sW pH pW]
    kernel_h = int(sys.argv[5])  # kH
    kernel_w = int(sys.argv[6])  # kW
    
    if len(sys.argv) >= 9:
        stride_h = int(sys.argv[7])  # sH
        stride_w = int(sys.argv[8])  # sW
        
    if len(sys.argv) >= 11:
        pad_h = int(sys.argv[9])   # pH
        pad_w = int(sys.argv[10])  # pW
else:
    print("Using default MaxPool parameters: kernel=2x2, stride=2x2, pad=0x0")

print(f"Creating MaxPool ONNX model: kernel=({kernel_h},{kernel_w}), stride=({stride_h},{stride_w}), pad=({pad_h},{pad_w})")

# Define input and output tensors with dynamic shapes
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None]) 

# Create MaxPool node with dynamic parameters
maxpool_node = helper.make_node(
    "MaxPool",
    inputs=["input"],
    outputs=["output"],
    kernel_shape=[kernel_h, kernel_w],
    strides=[stride_h, stride_w],
    pads=[pad_h, pad_w, pad_h, pad_w] if (pad_h > 0 or pad_w > 0) else None
)

# Remove pads attribute if no padding is needed
if pad_h == 0 and pad_w == 0:
    # Don't add pads attribute for zero padding
    maxpool_node = helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        kernel_shape=[kernel_h, kernel_w],
        strides=[stride_h, stride_w]
    )

graph = helper.make_graph(
    nodes=[maxpool_node],
    name="MaxPoolGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="maxpool_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "maxpool.onnx")

onnx.save(model, output_path)
print(f"MaxPool ONNX model saved to: {output_path}")