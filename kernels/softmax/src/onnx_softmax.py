import os
import sys
import onnx
from onnx import helper, TensorProto

# For Softmax, we expect a 2D tensor of shape [CHANNELS, INNER_SIZE]
# We'll define the input shape as [None, None] to be flexible
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None]) 

# Softmax node. axis=0 means softmax is applied along the 0-th (channels) dimension.
softmax_node = helper.make_node(
    "Softmax",
    inputs=["input"],
    outputs=["output"],
    axis=0  # Apply softmax along the first dimension (channels)
)

graph = helper.make_graph(
    nodes=[softmax_node],
    name="SoftmaxGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="softmax_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
output_path = os.path.join(OUTPUT_DIR, "softmax.onnx")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

onnx.save(model, output_path)