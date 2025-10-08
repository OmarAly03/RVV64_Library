import onnx
from onnx import helper, TensorProto
import os

# Configuration
KERNEL_SIZE = 3
STRIDE = 2
OUTPUT_DIR = "output_files"
MODEL_NAME = "maxpool.onnx"

# Define tensor shapes (using None for dynamic dimensions)
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ['N', 'C', 'H', 'W'])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ['N', 'C', 'OH', 'OW'])
I = helper.make_tensor_value_info("I", TensorProto.INT64, ['N', 'C', 'OH', 'OW']) # <-- ADDED INDICES TENSOR

# Create the MaxPool node, now specifying TWO outputs
node_def = helper.make_node(
    'MaxPool',
    ['X'],
    ['Y', 'I'], # <-- SPECIFY BOTH OUTPUTS HERE
    kernel_shape=[KERNEL_SIZE, KERNEL_SIZE],
    strides=[STRIDE, STRIDE],
)

# Create the graph, listing both Y and I as graph outputs
graph_def = helper.make_graph(
    [node_def],
    "maxpool-graph",
    [X],
    [Y, I], # <-- LIST BOTH OUTPUTS HERE
)

# Create the model, specifying an older opset and IR version for compatibility
model_def = helper.make_model(graph_def, producer_name="gemini-maxpool-producer", opset_imports=[helper.make_opsetid("", 11)])
model_def.ir_version = 7 # A widely compatible IR version

os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx.save(model_def, os.path.join(OUTPUT_DIR, MODEL_NAME))
print(f"Saved ONNX model to {os.path.join(OUTPUT_DIR, MODEL_NAME)}")
