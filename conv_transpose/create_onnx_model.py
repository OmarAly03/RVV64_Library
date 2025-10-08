import numpy as np
import onnx
from onnx import helper, TensorProto
import os

def create_transposed_conv2d_model(input_shape, kernel_shape, output_channels,
                                   stride=[1, 1], padding=[0, 0], output_padding=[0, 0], use_bias=False):

    batch, in_channels, h, w = input_shape
    kernel_in_ch, kernel_out_ch, kh, kw = kernel_shape

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, kernel_shape)

    # Calculate output shape
    out_h = (h - 1) * stride[0] - 2 * padding[0] + kh + output_padding[0]
    out_w = (w - 1) * stride[1] - 2 * padding[1] + kw + output_padding[1]
    output_shape = [batch, output_channels, out_h, out_w]

    # Define output
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    initializers = []
    node_inputs = ['X', 'W']

    if use_bias:
        b = helper.make_tensor(
            name='b',
            data_type=TensorProto.FLOAT,
            dims=[output_channels],
            vals=np.random.randn(output_channels).flatten().astype(np.float32)
        )
        initializers.append(b)
        node_inputs.append('b')

    # Create ConvTranspose node
    conv_transpose_node = helper.make_node(
        'ConvTranspose',
        inputs=node_inputs,
        outputs=['Y'],
        kernel_shape=[kh, kw],
        strides=stride,
        pads=padding + padding,  # [top, left, bottom, right]
        output_padding=output_padding,
        name='transposed_conv'
    )

    graph = helper.make_graph(
        [conv_transpose_node],
        'transposed_conv_graph',
        [X, W],  # Model inputs
        [Y],     # Model outputs
        initializers
    )

    # Create model
    model = helper.make_model(graph, producer_name='onnx-transposed-conv')
    model.opset_import[0].version = 11
    
    model.ir_version = 7 # <-- Added this as a fix for onnx

    # Check model validity
    onnx.checker.check_model(model)

    return model

if __name__ == "__main__":
    import sys
    
    # Default parameters
    batch_size = 1
    in_channels = 1
    out_channels = 1
    input_h, input_w = 4, 4
    kernel_h, kernel_w = 3, 3
    stride = [2, 2]
    padding = [1, 1]

    # Parse command line arguments if provided
    if len(sys.argv) >= 7:
        input_h = input_w = int(sys.argv[1])
        kernel_h = kernel_w = int(sys.argv[2])
        stride_val = int(sys.argv[3])
        stride = [stride_val, stride_val]
        padding_val = int(sys.argv[4])
        padding = [padding_val, padding_val]
        in_channels = int(sys.argv[5])
        out_channels = int(sys.argv[6])
    elif len(sys.argv) >= 2:
        input_h = input_w = int(sys.argv[1])

    # Define shapes for inputs
    input_shape = (batch_size, in_channels, input_h, input_w)
    kernel_shape = (in_channels, out_channels, kernel_h, kernel_w)

    # Create the ONNX model
    model = create_transposed_conv2d_model(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        output_channels=out_channels,
        stride=stride,
        padding=padding
    )

    # Save the model
    output_dir = "./output_files"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "conv_transpose.onnx")
    onnx.save(model, model_path)
    
    print(f"ONNX model saved to: {model_path}")
