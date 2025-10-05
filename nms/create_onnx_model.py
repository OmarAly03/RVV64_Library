import onnx
import os
from src.onnx_nms import create_onnx_nms_model

if __name__ == "__main__":
    output_dir = "./output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = create_onnx_nms_model()
    model.ir_version = 10
    onnx.save(model, os.path.join(output_dir, "nms.onnx"))