import onnx
import os
import sys
from src.onnx_nms import create_onnx_nms_model

if __name__ == "__main__":
    output_dir = "./output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(sys.argv) >= 4:
        max_output_boxes_per_class = int(sys.argv[1])
        iou_threshold = float(sys.argv[2])
        score_threshold = float(sys.argv[3])
    else:
        max_output_boxes_per_class = 50
        iou_threshold = 0.5
        score_threshold = 0.1

    model = create_onnx_nms_model(max_output_boxes_per_class, iou_threshold, score_threshold)
    model.ir_version = 10
    onnx.save(model, os.path.join(output_dir, "nms.onnx"))