#!/usr/bin/env python3
import sys
import numpy as np

def emit_val(name, val):
    print(f".global {name}\n.section .data\n.balign 4\n{name}:\n    .word {val}")

def emit_array(name, array, alignment='256'):
    print(f".global {name}\n.section .data\n.balign {alignment}\n{name}:")
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += f"{bs[i+3-n]:02x}" if i+3-n < len(bs) else "00"
        print(f"    .word 0x{s}")

def dense_reference(input_data, weights, bias, IN, OUT):
    # Weights shape: [OUT, IN]
    w_mat = weights.reshape(OUT, IN)
    return np.dot(w_mat, input_data) + bias

if __name__ == "__main__":
    # Default F4: 120 in, 84 out
    args = [120, 84]
    if len(sys.argv) >= 3:
        args = [int(x) for x in sys.argv[1:3]]
        
    IN, OUT = args
    
    np.random.seed(42)
    input_data = np.random.rand(IN).astype(np.float32)
    weights = np.random.rand(OUT * IN).astype(np.float32)
    bias = np.random.rand(OUT).astype(np.float32)
    golden = dense_reference(input_data, weights, bias, IN, OUT)
    
    print(".section .data,\"aw\",@progbits")
    emit_val("IN_DIM", IN)
    emit_val("OUT_DIM", OUT)
    
    emit_array("input_data", input_data)
    emit_array("weights", weights)
    emit_array("bias", bias)
    emit_array("golden_data", golden)
    # Placeholder
    emit_array("output_data", np.zeros(OUT, dtype=np.float32))