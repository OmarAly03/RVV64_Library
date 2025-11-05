import os
import onnx
from onnx import helper, TensorProto

# A: [M, K] matrix, B: [K, N] matrix, C: [M, N] result matrix
a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None, None])  # [M, K]
b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, None])  # [K, N]
c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [None, None])  # [M, N]

matmul_node = helper.make_node(
    "MatMul",     # operator name
    inputs=["a", "b"],
    outputs=["c"]
)

graph = helper.make_graph(
    nodes=[matmul_node],
    name="MatrixMultiplyGraph",
    inputs=[a, b],
    outputs=[c]
)

# Creating the model
model = helper.make_model(graph, producer_name="matrix_multiply_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
output_path = os.path.join(OUTPUT_DIR, "matrix_multiply.onnx")

onnx.save(model, output_path)