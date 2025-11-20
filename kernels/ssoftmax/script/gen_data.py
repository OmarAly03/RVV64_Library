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

def softmax_ref(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    # Args: Size
    if len(sys.argv) < 2:
        SIZE = 10 # Default LeNet F5
    else:
        SIZE = int(sys.argv[1])

    print(".section .data,\"aw\",@progbits")
    print(f".global SIZE; SIZE: .word {SIZE}")

    np.random.seed(42)
    inp = np.random.rand(SIZE).astype(np.float32)
    gold = softmax_ref(inp)

    emit_array("input_data", inp)
    emit_array("golden_data", gold)
    emit_array("output_data", np.zeros(SIZE, dtype=np.float32))