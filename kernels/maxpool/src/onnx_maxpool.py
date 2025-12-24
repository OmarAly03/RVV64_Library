import os
import onnx
from onnx import helper, TensorProto

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None]) 

maxpool_node = helper.make_node(
    "MaxPool",  # operator name
    inputs=["input"],
    outputs=["output"],
    kernel_shape=[2, 2],
    strides=[2, 2]
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