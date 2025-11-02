import onnx
from onnx import helper, TensorProto
import os

def create_transposed_conv2d_model(input_shape, kernel_shape, output_channels,
                                   stride=[1, 1], padding=[0, 0]):

    batch, in_channels, h, w = input_shape
    kernel_in_ch, kernel_out_ch, kh, kw = kernel_shape

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, kernel_shape)

    out_h = (h - 1) * stride[0] - 2 * padding[0] + kh
    out_w = (w - 1) * stride[1] - 2 * padding[1] + kw
    output_shape = [batch, output_channels, out_h, out_w]

    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    conv_transpose_node = helper.make_node(
        'ConvTranspose',
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[kh, kw],
        strides=stride,
        pads=padding + padding,
        name='transposed_conv'
    )

    graph = helper.make_graph(
        [conv_transpose_node],
        'transposed_conv_graph',
        [X, W],
        [Y],
        []
    )

    model = helper.make_model(graph, producer_name='onnx-transposed-conv')
    model.opset_import[0].version = 11
    model.ir_version = 7

    onnx.checker.check_model(model)

    return model

if __name__ == "__main__":
    import sys
    
    batch_size = 1
    in_channels = 1
    out_channels = 1
    input_h, input_w = 4, 4
    kernel_h, kernel_w = 3, 3
    stride = [2, 2]
    padding = [0, 0]

    if len(sys.argv) >= 5:
        input_h = input_w = int(sys.argv[1])
        in_channels = int(sys.argv[2])
        out_channels = int(sys.argv[3])
        padding_val = int(sys.argv[4])
        padding = [padding_val, padding_val]
    elif len(sys.argv) >= 4:
        input_h = input_w = int(sys.argv[1])
        in_channels = int(sys.argv[2])
        out_channels = int(sys.argv[3])
    elif len(sys.argv) >= 2:
        input_h = input_w = int(sys.argv[1])

    input_shape = (batch_size, in_channels, input_h, input_w)
    kernel_shape = (in_channels, out_channels, kernel_h, kernel_w)

    model = create_transposed_conv2d_model(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        output_channels=out_channels,
        stride=stride,
        padding=padding
    )

    output_dir = "./output_files"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "conv_transpose.onnx")
    onnx.save(model, model_path)
    
    print(f"ONNX model saved to: {model_path}")
