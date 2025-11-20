#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# SCRIPT Adapted for Im2Col by Gemini

import numpy as np
import sys

def emit_val(name, val):
    print(".global %s" % name)
    print(".balign 8")
    print("%s:" % name)
    print("    .word 0x%08x" % (val & 0xFFFFFFFF))
    print("    .word 0x%08x" % (val >> 32))

def emit_array(name, array, alignment='256'):
    print(".global %s" % name)
    print(".balign " + alignment)
    print("%s:" % name)
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += "%02x" % bs[i+3-n]
        print("    .word 0x%s" % s)

def conv2d_reference(input_data, weights, bias, C, H, W, M, KH, KW, pad, stride):
    # Naive python loop to match C++ reference exactly
    pad_h, pad_w = pad, pad
    stride_h, stride_w = stride, stride
    
    out_h = (H + 2*pad_h - KH) // stride_h + 1
    out_w = (W + 2*pad_w - KW) // stride_w + 1
    
    output = np.zeros((M, out_h, out_w), dtype=np.float32)
    
    # Assuming input is flat in the C++ code, we treat it as such here
    # Input: C, H, W
    input_reshaped = input_data.reshape(C, H, W)
    weights_reshaped = weights.reshape(M, C, KH, KW)
    
    for m in range(M):
        for oh in range(out_h):
            for ow in range(out_w):
                sum_val = bias[m] if len(bias) > 0 else 0.0
                for c in range(C):
                    for kh in range(KH):
                        ih = oh * stride_h + kh - pad_h
                        for kw in range(KW):
                            iw = ow * stride_w + kw - pad_w
                            if 0 <= ih < H and 0 <= iw < W:
                                sum_val += input_reshaped[c, ih, iw] * weights_reshaped[m, c, kh, kw]
                output[m, oh, ow] = sum_val
    return output.flatten()

if len(sys.argv) == 10:
    # C H W M KH KW PAD STRIDE HAS_BIAS
    C = int(sys.argv[1])
    H = int(sys.argv[2])
    W = int(sys.argv[3])
    M = int(sys.argv[4])
    KH = int(sys.argv[5])
    KW = int(sys.argv[6])
    PAD = int(sys.argv[7])
    STRIDE = int(sys.argv[8])
    HAS_BIAS = int(sys.argv[9])
else:
    print("Error args: C H W M KH KW PAD STRIDE HAS_BIAS")
    sys.exit()

dtype = np.float32

input_data = np.random.rand(C * H * W).astype(dtype)
weights = np.random.rand(M * C * KH * KW).astype(dtype)
bias = np.random.rand(M).astype(dtype) if HAS_BIAS else np.zeros(M).astype(dtype)

golden = conv2d_reference(input_data, weights, bias, C, H, W, M, KH, KW, PAD, STRIDE)
out_sz = golden.shape[0]

# Calculate derived constants for C++ buffer sizing
OUT_H = (H + 2*PAD - KH) // STRIDE + 1
OUT_W = (W + 2*PAD - KW) // STRIDE + 1
K_DIM = C * KH * KW
N_DIM = OUT_H * OUT_W
COL_SIZE = K_DIM * N_DIM
OUT_SIZE = M * OUT_H * OUT_W

print(".section .data,\"aw\",@progbits")
emit_val("C_IN", C)
emit_val("H_IN", H)
emit_val("W_IN", W)
emit_val("M_OUT", M)
emit_val("K_H", KH)
emit_val("K_W", KW)
emit_val("PAD", PAD)
emit_val("STRIDE", STRIDE)
emit_val("HAS_BIAS", HAS_BIAS)

# Debug/Check constants
emit_val("CHK_COL_SIZE", COL_SIZE)
emit_val("CHK_OUT_SIZE", OUT_SIZE)

emit_array("input_data", input_data, '256')
emit_array("weights", weights, '256')
emit_array("bias", bias, '256')
emit_array("golden", golden, '256')
# Placeholders
emit_array("output_data", np.zeros(OUT_SIZE, dtype=dtype), '256')