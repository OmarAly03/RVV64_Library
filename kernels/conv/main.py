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


def build_dynamic_conv_model(weight, bias, kH, kW, sH, sW, pH, pW):
    """Build an ONNX Conv model programmatically with provided weights/bias.
    Inputs:  input [N,C,H,W]
    Outputs: output [N,Cout,Hout,Wout]
    Weights provided as initializers so session only needs the input tensor.
    """
    from onnx import helper, TensorProto

    N = None  # dynamic
    Cin = weight.shape[1]
    Cout = weight.shape[0]

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, Cin, None, None])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, Cout, None, None])

    W_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=list(weight.shape),
        vals=weight.astype(np.float32).flatten().tolist(),
    )

    if bias is None:
        bias = np.zeros((Cout,), dtype=np.float32)
    B_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[Cout],
        vals=bias.astype(np.float32).flatten().tolist(),
    )

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        kernel_shape=[kH, kW],
        strides=[sH, sW],
        pads=[pH, pW, pH, pW],
        dilations=[1, 1],
        group=1,
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name="ConvGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W_init, B_init],
    )
    model = helper.make_model(graph, producer_name="conv_dynamic", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def run():
    # Default problem size
    N, Cin, Cout, H, W = 1, 3, 3, 8, 8
    kH, kW = 3, 3
    sH, sW = 1, 1
    pH, pW = 1, 1

    # Parse CLI like run_conv2d.cpp: N Cin Cout H W [kH kW sH sW pH pW]
    args = sys.argv[1:]
    if len(args) >= 5:
        N, Cin, Cout, H, W = map(int, args[:5])
    if len(args) >= 7:
        kH, kW = map(int, args[5:7])
    if len(args) >= 9:
        sH, sW = map(int, args[7:9])
    if len(args) >= 11:
        pH, pW = map(int, args[9:11])

    # Paths to I/O binaries
    input_path = os.path.join(OUT_DIR, "input.bin")
    kernel_path = os.path.join(OUT_DIR, "kernel.bin")

    # If input/kernel not present yet, run the RVV runner first to generate them
    # However, to keep parity with matmul workflow, we expect Makefile run to have already
    # executed the RVV binary which writes these files. But we also handle the case they're missing.
    if not (os.path.exists(input_path) and os.path.exists(kernel_path)):
        runner = os.path.join(OUT_DIR, "run_conv2d")
        cmd = ["qemu-riscv64", "-cpu", "rv64,v=true", runner]
        # match argument contract
        cmd += list(map(str, [N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW]))
        subprocess.run(cmd, check=True)

    # Load inputs
    A = np.fromfile(input_path, dtype=np.float32).reshape(N, Cin, H, W)
    K = np.fromfile(kernel_path, dtype=np.float32).reshape(Cout, Cin, kH, kW)

    # Build and run ONNX Conv with K as the initializer
    model = build_dynamic_conv_model(K, bias=None, kH=kH, kW=kW, sH=sH, sW=sW, pH=pH, pW=pW)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString())
    onnx_ref = session.run(None, {session.get_inputs()[0].name: A.astype(np.float32)})[0]

    # Compute derived output shape
    outH = (H + 2 * pH - kH) // sH + 1
    outW = (W + 2 * pW - kW) // sW + 1

    # Load C outputs
    def load(name):
        path = os.path.join(OUT_DIR, name)
        return np.fromfile(path, dtype=np.float32).reshape(N, Cout, outH, outW)

    c_scalar = load("c_scalar.bin")
    c_e32m1 = load("c_e32m1.bin")
    c_e32m2 = load("c_e32m2.bin")
    c_e32m4 = load("c_e32m4.bin")
    c_e32m8 = load("c_e32m8.bin")
    c_im2col_gemm_m8 = load("c_im2col_gemm_m8.bin")

    # Results table
    implementations = [
        ("ONNX Golden Ref", onnx_ref),
        ("C Scalar", c_scalar),
        ("C Vectorized (e32m1)", c_e32m1),
        ("C Vectorized (e32m2)", c_e32m2),
        ("C Vectorized (e32m4)", c_e32m4),
        ("C Vectorized (e32m8)", c_e32m8),
        ("C Im2Col+GEMM (m8)", c_im2col_gemm_m8),
    ]

    print(f"\nConv2D: N={N} Cin={Cin} Cout={Cout} HxW={H}x{W} k={kH}x{kW} stride=({sH},{sW}) pad=({pH},{pW})")
    total_ops = 1.0 * N * Cout * outH * outW * Cin * kH * kW * 2
    print(f"Total operations: {int(total_ops):,} FLOPs")

    ref = onnx_ref
    print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
    print("-" * 60)
    for name, result in implementations:
        mae = max_abs_error(ref, result)
        snr = snr_db(ref, result)
        print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")


if __name__ == "__main__":
    run()
