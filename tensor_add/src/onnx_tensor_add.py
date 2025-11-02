import os
import onnx
from onnx import helper, TensorProto

# A: input_a [size]
input_tensor_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None]) 
# B: input_b [size]
input_tensor_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [None]) 

# Y: output [size]
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None]) 

# 'Add' node. No broadcasting is needed since both inputs have the same shape.
add_node = helper.make_node(
    "Add",
    inputs=["A", "B"],
    outputs=["Y"]
)

graph = helper.make_graph(
    nodes=[add_node],
    name="TensorAddGraph",
    inputs=[input_tensor_a, input_tensor_b],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="tensor_add_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
output_path = os.path.join(OUTPUT_DIR, "tensor_add.onnx")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

onnx.save(model, output_path)