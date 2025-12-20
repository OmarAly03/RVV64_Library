#!/usr/bin/env python3

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, ValueInfoProto
import tempfile
import os

def create_onnx_nms_model(max_output_boxes_per_class=50, iou_threshold=0.5, score_threshold=0.1, center_point_box=0):
    """
    Create an ONNX model with NonMaxSuppression operator
    
    Args:
        max_output_boxes_per_class: Maximum number of boxes per class
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering
        center_point_box: Box format (0=corner, 1=center)
    
    Returns:
        ONNX model
    """
    
    # Define input shapes (dynamic batch size and spatial dimension)
    boxes_shape = ['batch_size', 'spatial_dimension', 4]
    scores_shape = ['batch_size', 'num_classes', 'spatial_dimension']
    
    # Input definitions
    boxes_input = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, boxes_shape)
    scores_input = helper.make_tensor_value_info('scores', TensorProto.FLOAT, scores_shape)
    
    # Create constant inputs for NMS parameters
    max_output_boxes_const = helper.make_tensor(
        name='max_output_boxes_per_class',
        data_type=TensorProto.INT64,
        dims=[],
        vals=[max_output_boxes_per_class]
    )
    
    iou_threshold_const = helper.make_tensor(
        name='iou_threshold', 
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[iou_threshold]
    )
    
    score_threshold_const = helper.make_tensor(
        name='score_threshold',
        data_type=TensorProto.FLOAT, 
        dims=[],
        vals=[score_threshold]
    )
    
    # Output definition
    output = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, ['num_selected_boxes', 3])
    
    # Create NonMaxSuppression node
    nms_node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        center_point_box=center_point_box
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[nms_node],
        name='NMSGraph',
        inputs=[boxes_input, scores_input],
        outputs=[output],
        initializer=[
            max_output_boxes_const,
            iou_threshold_const, 
            score_threshold_const
        ]
    )
    
    # Create the model
    opset_imports = [helper.make_opsetid("", 22)]
    model = helper.make_model(graph, producer_name='nms_test', opset_imports=opset_imports)
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

def test_onnx_nms(boxes, scores, max_output_boxes_per_class=50, iou_threshold=0.5, score_threshold=0.1, center_point_box=0):
    """
    Test NMS using ONNX runtime
    
    Args:
        boxes: Box coordinates array [batch_size, spatial_dimension, 4]
        scores: Score array [batch_size, num_classes, spatial_dimension]
        max_output_boxes_per_class: Maximum boxes per class
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        center_point_box: Box format
        
    Returns:
        Selected indices array [num_selected, 3]
    """
    
    # Ensure inputs are float32
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    
    print(f"ONNX NMS input shapes: boxes={boxes.shape}, scores={scores.shape}")
    
    # Create ONNX model
    model = create_onnx_nms_model(
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        center_point_box=center_point_box
    )
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx.save(model, tmp_file.name)
        model_path = tmp_file.name
    
    try:
        # Create inference session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Run inference
        input_dict = {
            'boxes': boxes,
            'scores': scores
        }
        
        outputs = session.run(['selected_indices'], input_dict)
        selected_indices = outputs[0]
        
        print(f"ONNX NMS selected {selected_indices.shape[0]} boxes")
        
        return selected_indices
        
    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)

def validate_nms_implementation(our_results, onnx_results, tolerance=1e-6):
    """
    Validate our NMS implementation against ONNX reference
    
    Args:
        our_results: Results from our implementation [num_selected, 3]
        onnx_results: Results from ONNX [num_selected, 3] 
        tolerance: Numerical tolerance
        
    Returns:
        dict with validation metrics
    """
    
    # Convert to sets for exact comparison (indices should be integers)
    our_set = set(map(tuple, our_results))
    onnx_set = set(map(tuple, onnx_results))
    
    # Compute intersection and differences
    intersection = our_set.intersection(onnx_set)
    our_only = our_set - onnx_set
    onnx_only = onnx_set - our_set
    
    # Compute metrics
    total_our = len(our_set)
    total_onnx = len(onnx_set)
    matches = len(intersection)
    
    precision = matches / total_our if total_our > 0 else 0.0
    recall = matches / total_onnx if total_onnx > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_our': total_our,
        'total_onnx': total_onnx,
        'matches': matches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'our_only': our_only,
        'onnx_only': onnx_only,
        'exact_match': len(our_only) == 0 and len(onnx_only) == 0
    }

if __name__ == "__main__":
    # Simple test
    print("Testing ONNX NMS implementation...")
    
    # Create test data
    batch_size = 1
    num_classes = 2
    spatial_dimension = 10
    
    np.random.seed(42)
    boxes = np.random.rand(batch_size, spatial_dimension, 4).astype(np.float32) * 100
    scores = np.random.rand(batch_size, num_classes, spatial_dimension).astype(np.float32)
    
    # Fix box format to ensure y1 < y2, x1 < x2
    for i in range(spatial_dimension):
        y1, x1, y2, x2 = boxes[0, i]
        boxes[0, i] = [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
    
    try:
        selected_indices = test_onnx_nms(boxes, scores)
        print(f"ONNX test completed successfully. Selected {selected_indices.shape[0]} boxes.")
        print(f"First few selected indices:")
        for i in range(min(5, selected_indices.shape[0])):
            print(f"  [{selected_indices[i, 0]}, {selected_indices[i, 1]}, {selected_indices[i, 2]}]")
    except Exception as e:
        print(f"ONNX test failed: {e}")