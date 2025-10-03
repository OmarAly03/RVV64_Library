import onnx
from onnx import helper
from onnx import TensorProto
import sys
import os

def create_nms_model(output_path, N):
    # Define the input tensor, note the shape for 1D MaxPool
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, N])

    # Define the output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, N])

    # Define the MaxPool node
    node_def = helper.make_node(
        'MaxPool',
        ['X'],
        ['P'],
        kernel_shape=[3],
        strides=[1],
        pads=[1, 1]
    )
    
    # Define the comparison logic using Less, Greater, Or, Not
    less_node = helper.make_node('Less', ['X', 'P'], ['Less_out'])
    greater_node = helper.make_node('Greater', ['X', 'P'], ['Greater_out'])
    or_node = helper.make_node('Or', ['Less_out', 'Greater_out'], ['Or_out'])
    not_node = helper.make_node('Not', ['Or_out'], ['M'])


    # Define the cast node to convert boolean mask to float
    cast_node = helper.make_node(
        'Cast',
        ['M'],
        ['C'],
        to=TensorProto.FLOAT
    )

    # Define the multiplication node to apply the mask
    mul_node = helper.make_node(
        'Mul',
        ['X', 'C'],
        ['Y']
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def, less_node, greater_node, or_node, not_node, cast_node, mul_node],
        'nms-model',
        [X],
        [Y],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='onnx-nms', ir_version=7)
    model_def.opset_import[0].version = 10

    onnx.save(model_def, output_path)

if __name__ == '__main__':
    N = 16
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
        
    # Correctly resolve the output path relative to the script's location
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, '..', 'output_files', 'nms.onnx')
    create_nms_model(output_path, N)