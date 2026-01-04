import os
import onnx
from onnx import helper, TensorProto

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None]) 
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None]) 

leaky_relu_node = helper.make_node(
    "LeakyRelu",     # operator name
    inputs=["input"],
    outputs=["output"],
    alpha=0.01       # negative slope parameter (default is 0.01)
)

graph = helper.make_graph(
    nodes=[leaky_relu_node],
    name="LeakyReLUActivationGraph",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="leaky_relu_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "leaky_relu.onnx")

onnx.save(model, output_path)
print(f"LeakyReLU ONNX model saved to: {output_path}")