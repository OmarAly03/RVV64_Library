import os
import onnx
from onnx import helper, TensorProto

# Input tensors with dynamic shapes
data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, [None, None]) 
indices_tensor = helper.make_tensor_value_info("indices", TensorProto.INT64, [None, None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None]) 

# GatherElements node with axis=0 as default
gather_node = helper.make_node(
    "GatherElements",
    inputs=["data", "indices"],
    outputs=["output"],
    axis=0
)

graph = helper.make_graph(
    nodes=[gather_node],
    name="GatherElementsGraph",
    inputs=[data_tensor, indices_tensor],
    outputs=[output_tensor]
)

# Use opset 11 for GatherElements
model = helper.make_model(graph, producer_name="gather_elements_example", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "gather_elements.onnx")

onnx.save(model, output_path)
