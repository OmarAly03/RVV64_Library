import os
import onnx
from onnx import helper, TensorProto

# Define tensor shapes. Using None for flexibility (e.g., batch_size)
# A: input [batch_size, in_features]
input_tensor = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]) 
# B: weights [out_features, in_features]
weights_tensor = helper.make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
# C: bias [out_features]
bias_tensor = helper.make_tensor_value_info("C", TensorProto.FLOAT, [None]) 

# Y: output [batch_size, out_features]
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None]) 

# Gemm node implements Y = alpha*A*B + beta*C
# We want Y = 1.0*A*B^T + 1.0*C
# B shape is [out_features, in_features], so B^T is [in_features, out_features]
# A shape is [batch_size, in_features]
# A * B^T gives [batch_size, out_features], which is correct.
gemm_node = helper.make_node(
    "Gemm",
    inputs=["A", "B", "C"],
    outputs=["Y"],
    alpha=1.0,
    beta=1.0,
    transB=1  # This is crucial! It transposes B before multiplication.
)

graph = helper.make_graph(
    nodes=[gemm_node],
    name="DenseGraph",
    inputs=[input_tensor, weights_tensor, bias_tensor],
    outputs=[output_tensor]
)

model = helper.make_model(graph, producer_name="dense_example", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_files")
output_path = os.path.join(OUTPUT_DIR, "dense.onnx")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

onnx.save(model, output_path)