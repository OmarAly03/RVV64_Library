import onnx
from onnx import helper, TensorProto
import numpy as np

# Input tensor: [N, C_in, H, W] - batch, input channels, height, width
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None, None, None])

# Output tensor: [N, C_out, H_out, W_out]
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])

# Weight tensor: [C_out, C_in, kernel_H, kernel_W]
# For example: 64 output channels, 3 input channels, 3x3 kernel
weight_shape = [64, 3, 3, 3]
weight_data = np.random.randn(*weight_shape).astype(np.float32)
weight_tensor = helper.make_tensor(
    name="weight",
    data_type=TensorProto.FLOAT,
    dims=weight_shape,
    vals=weight_data.flatten().tolist()
)

# Optional: Bias tensor [C_out]
bias_shape = [64]
bias_data = np.zeros(bias_shape).astype(np.float32)
bias_tensor = helper.make_tensor(
    name="bias",
    data_type=TensorProto.FLOAT,
    dims=bias_shape,
    vals=bias_data.flatten().tolist()
)

# Create Conv node
conv_node = helper.make_node(
    "Conv",
    inputs=["input", "weight", "bias"],
    outputs=["output"],
    kernel_shape=[3, 3],      # Kernel size
    strides=[1, 1],           # Stride
    pads=[1, 1, 1, 1],        # Padding [top, left, bottom, right]
    dilations=[1, 1],         # Dilation
    group=1                   # Number of groups (1 = standard conv)
)

# Create graph
graph = helper.make_graph(
    nodes=[conv_node],
    name="ConvolutionGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[weight_tensor, bias_tensor]  # Weights and biases as constants
)

# Create model
model = helper.make_model(
    graph, 
    producer_name="conv_example", 
    opset_imports=[helper.make_opsetid("", 13)]
)
model.ir_version = 7

# Save model
onnx.save(model, "./conv/output_files/convolution.onnx")
print("Saved convolution.onnx")