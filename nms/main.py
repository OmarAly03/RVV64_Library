#!/usr/bin/env python3

import numpy as np
import struct
import sys
import os
import onnx
import onnxruntime as ort

# =============== Import Utility Functions ===============
from src.onnx_utils import max_abs_error, snr_db
from src.nms_utils import load_binary_data, nms_reference_numpy

# ==== Loading ONNX Model ====
onnx_model = onnx.load("./output_files/nms.onnx")
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession("./output_files/nms.onnx")

def load_nms_results(filename):
    """Load NMS results from binary file"""
    results = []
    try:
        with open(filename, 'rb') as f:
            count_bytes = f.read(8)
            if len(count_bytes) < 8:
                return results
            
            count = struct.unpack('Q', count_bytes)[0]
            
            for _ in range(count):
                batch_bytes = f.read(8)
                class_bytes = f.read(8)
                box_bytes = f.read(8)
                
                if len(batch_bytes) < 8 or len(class_bytes) < 8 or len(box_bytes) < 8:
                    break
                    
                batch_idx = struct.unpack('q', batch_bytes)[0]
                class_idx = struct.unpack('q', class_bytes)[0]
                box_idx = struct.unpack('q', box_bytes)[0]
                
                results.append([batch_idx, class_idx, box_idx])
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    
    return np.array(results) if results else np.array([]).reshape(0, 3)

def main():
    # Parse command line arguments
    if len(sys.argv) >= 7:
        num_batches = int(sys.argv[1])
        num_classes = int(sys.argv[2])
        spatial_dimension = int(sys.argv[3])
        max_output_boxes_per_class = int(sys.argv[4])
        iou_threshold = float(sys.argv[5])
        score_threshold = float(sys.argv[6])
    else:
        num_batches = 1
        num_classes = 2
        spatial_dimension = 100
        max_output_boxes_per_class = 50
        iou_threshold = 0.5
        score_threshold = 0.1

    # Load input data
    boxes = load_binary_data(f"./output_files/boxes.bin", np.float32).reshape(num_batches, spatial_dimension, 4)
    scores = load_binary_data(f"./output_files/scores.bin", np.float32).reshape(num_batches, num_classes, spatial_dimension)

    # ==== ONNX Golden Reference (using ONNXRuntime) ====
    input_names = [input.name for input in onnx_model.graph.input]
    onnx_ref = session.run(None, {input_names[0]: boxes, input_names[1]: scores})[0]

    # ==== Python Scalar ====
    py_scalar = nms_reference_numpy(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

    # ==== C Implementations ====
    c_scalar = load_nms_results("./output_files/nms_scalar.bin")
    c_e32m1 = load_nms_results("./output_files/nms_e32m1.bin")
    c_e32m2 = load_nms_results("./output_files/nms_e32m2.bin")
    c_e32m4 = load_nms_results("./output_files/nms_e32m4.bin")
    c_e32m8 = load_nms_results("./output_files/nms_e32m8.bin")

    # ==== Results Table ====
    implementations = [
        ("ONNX Golden Ref", onnx_ref),
        ("Python Scalar", py_scalar),
        ("C Scalar", c_scalar),
        ("C Vectorized (e32m1)", c_e32m1),
        ("C Vectorized (e32m2)", c_e32m2),
        ("C Vectorized (e32m4)", c_e32m4),
        ("C Vectorized (e32m8)", c_e32m8),
    ]

    print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
    print("-" * 60)

    for name, result in implementations:
        ref_set = set(map(tuple, onnx_ref))
        res_set = set(map(tuple, result))
        
        intersection = len(ref_set.intersection(res_set))
        precision = intersection / len(res_set) if len(res_set) > 0 else 1.0
        recall = intersection / len(ref_set) if len(ref_set) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0

        mae = max_abs_error(f1_score)
        snr = snr_db(mae)

        print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")

if __name__ == "__main__":
    main()