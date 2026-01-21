import numpy as np
import onnx
import onnxruntime as ort
import sys
import os
import subprocess

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output_files")
os.makedirs(OUT_DIR, exist_ok=True)

# Utils from this module
from src.onnx_utils import max_abs_error, snr_db


def build_3x3_conv_model(kernel, pad_h, pad_w):
    """Build an ONNX Conv model for 3x3 single-channel convolution.
    Inputs:  input [1, 1, H, W]
    Outputs: output [1, 1, Hout, Wout]
    """
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, None, None])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, None, None])

    # Reshape kernel from 3x3 to 1x1x3x3 (OIHW format)
    kernel_oihw = kernel.reshape(1, 1, 3, 3)
    
    W_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=list(kernel_oihw.shape),
        vals=kernel_oihw.astype(np.float32).flatten().tolist(),
    )

    # No bias for fair comparison
    bias = np.zeros(1, dtype=np.float32)
    B_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=bias.astype(np.float32).flatten().tolist(),
    )

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[pad_h, pad_w, pad_h, pad_w],
        dilations=[1, 1],
        group=1,
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name="Conv3x3Graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W_init, B_init],
    )
    model = helper.make_model(graph, producer_name="conv3x3", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def run():
    # Default: 128x128 with 3x3 kernel and padding=1
    H = 128
    W = 128
    pad_h = 1
    pad_w = 1
    use_padding = 1

    # Parse CLI: H W [pad_h pad_w use_padding]
    args = sys.argv[1:]
    if len(args) >= 2:
        H, W = map(int, args[:2])
    if len(args) >= 4:
        pad_h, pad_w = map(int, args[2:4])
    if len(args) >= 5:
        use_padding = int(args[4])

    # Paths to I/O binaries
    input_path = os.path.join(OUT_DIR, "input_3x3.bin")
    kernel_path = os.path.join(OUT_DIR, "kernel_3x3.bin")

    # If input/kernel not present yet, run the C++ runner first
    if not (os.path.exists(input_path) and os.path.exists(kernel_path)):
        runner = os.path.join(OUT_DIR, "run_conv2d_3x3")
        cmd = ["qemu-riscv64", "-cpu", "rv64,v=true", runner]
        cmd += list(map(str, [H, W, 3, 3, 1, 1, pad_h, pad_w, use_padding]))
        print(f"Running C++ test: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Load inputs
    input_data = np.fromfile(input_path, dtype=np.float32).reshape(H, W)
    kernel_data = np.fromfile(kernel_path, dtype=np.float32).reshape(3, 3)

    # Reshape for ONNX (add batch and channel dims)
    input_onnx = input_data.reshape(1, 1, H, W)

    # Compute output size
    if use_padding:
        out_h = H
        out_w = W
    else:
        out_h = H - 2
        out_w = W - 2

    # Build and run ONNX Conv
    model = build_3x3_conv_model(kernel_data, pad_h, pad_w)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString())
    onnx_ref = session.run(None, {session.get_inputs()[0].name: input_onnx.astype(np.float32)})[0]
    onnx_ref = onnx_ref.reshape(out_h, out_w)

    # Load C++ outputs
    def load(name):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            return np.fromfile(path, dtype=np.float32).reshape(out_h, out_w)
        return None

    c_m1 = load("c_3x3_m1.bin")
    c_m2 = load("c_3x3_m2.bin")
    c_m4 = load("c_3x3_m4.bin")
    c_m8 = load("c_3x3_m8.bin")
    c_m2_batched = load("c_3x3_m2_batched.bin")
    c_m4_batched = load("c_3x3_m4_batched.bin")
    c_m8_batched = load("c_3x3_m8_batched.bin")

    # Results table
    implementations = [
        ("ONNX (Reference)", onnx_ref),
        ("M1 (non-batched)", c_m1),
        ("M2 (non-batched)", c_m2),
        ("M4 (non-batched)", c_m4),
        ("M8 (non-batched)", c_m8),
        ("M2 (batched)", c_m2_batched),
        ("M4 (batched)", c_m4_batched),
        ("M8 (batched)", c_m8_batched),
    ]

    print(f"\n3x3 Conv2D: Input HxW={H}x{W}, Kernel=3x3, Pad=({pad_h},{pad_w}), use_padding={use_padding}")
    print(f"Output: {out_h}x{out_w}")
    total_ops = out_h * out_w * 3 * 3 * 2  # H * W * kernel_area * 2 (mult + add)
    print(f"Total operations: {int(total_ops):,} FLOPs\n")

    ref = onnx_ref
    print(f"{'Implementation':<25}{'Max Abs Error':<18}{'SNR (dB)':<15}{'Status':<10}")
    print("-" * 70)
    for name, result in implementations:
        if result is None:
            print(f"{name:<25}{'N/A':<18}{'N/A':<15}{'MISSING':<10}")
        else:
            mae = max_abs_error(ref, result)
            snr = snr_db(ref, result)
            status = "PASS" if mae < 1e-5 else "WARN" if mae < 1e-3 else "FAIL"
            print(f"{name:<25}{mae:<18.6g}{snr:<15.2f}{status:<10}")

    # Print value ranges for debugging
    print(f"\nValue Ranges:")
    print(f"ONNX Ref: [{np.min(onnx_ref):.6f}, {np.max(onnx_ref):.6f}]")
    if c_m1 is not None:
        print(f"M1:       [{np.min(c_m1):.6f}, {np.max(c_m1):.6f}]")
    if c_m8_batched is not None:
        print(f"M8 batch: [{np.min(c_m8_batched):.6f}, {np.max(c_m8_batched):.6f}]")


if __name__ == "__main__":
    run()
