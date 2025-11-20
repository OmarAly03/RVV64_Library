#!/usr/bin/env python3
import sys
import numpy as np

def emit_array(name, array, align='256'):
    print(f".global {name}\n.section .data\n.balign {align}\n{name}:")
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += f"{bs[i+3-n]:02x}" if i+3-n < len(bs) else "00"
        print(f"    .word 0x{s}")

if __name__ == "__main__":
    # Args: Batch, Channels, Height, Width
    if len(sys.argv) < 5:
        args = [1, 6, 28, 28] # Default LeNet C1
    else:
        args = [int(x) for x in sys.argv[1:5]]
        
    B, C, H, W = args

    print(".section .data,\"aw\",@progbits")
    print(f".global BATCH; BATCH: .word {B}")
    print(f".global CHANNELS; CHANNELS: .word {C}")
    print(f".global HEIGHT; HEIGHT: .word {H}")
    print(f".global WIDTH; WIDTH: .word {W}")

    np.random.seed(42)
    # Input: [B, C, H, W]
    inp = np.random.rand(B, C, H, W).astype(np.float32)
    # Bias: [C]
    bias = np.random.rand(C).astype(np.float32)
    
    # Golden Reference
    out = np.zeros_like(inp)
    for b in range(B):
        for c in range(C):
            out[b,c,:,:] = inp[b,c,:,:] + bias[c]

    emit_array("input_data", inp)
    emit_array("bias_data", bias)
    emit_array("golden_data", out)
    # Placeholder for output
    emit_array("output_data", np.zeros_like(out))