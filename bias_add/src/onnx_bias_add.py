import os
import onnx
from onnx import helper, TensorProto

# A: input [batch_size, channels, height, width]
input_tensor = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None, None, None]) 
# B: bias [1, channels, 1, 1]
bias_tensor = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, None, 1, 1])

# Y: output [batch_size, channels, height, width]
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None, None]) 

# 'Add' node with automatic broadcasting
add_node = helper.make_node(
    "Add",
    inputs=["A", "B"],
    outputs=["Y"]
)

graph = helper.make_graph(
    nodes=[add_node],
    name="BiasAddGraph",
    inputs=[input_tensor, bias_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="bias_add_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
output_path = os.path.join(OUTPUT_DIR, "bias_add.onnx")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

onnx.save(model, output_path)