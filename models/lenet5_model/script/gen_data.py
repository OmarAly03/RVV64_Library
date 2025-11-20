#!/usr/bin/env python3
import numpy as np
import sys
import os

def emit_array(name, array, align='256'):
    print(f".global {name}\n.section .data\n.balign {align}\n{name}:")
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += f"{bs[i+3-n]:02x}" if i+3-n < len(bs) else "00"
        print(f"    .word 0x{s}")

def load_or_rand(filename, shape):
    # try to load from ../data/filename
    path = os.path.join('data', filename)
    if os.path.exists(path):
        # sys.stderr.write(f"Loading {filename}...\n")
        return np.fromfile(path, dtype=np.float32)
    # else:
    #     # sys.stderr.write(f"Warning: {filename} not found, using random.\n")
    #     return np.random.rand(*shape).astype(np.float32)

if __name__ == "__main__":
    print(".section .data,\"aw\",@progbits")
    np.random.seed(42)

    # --- INPUT IMAGE (1x32x32) ---
    img = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/image_binaries/3.bin", (1 * 32 * 32,))
    emit_array("image_data", img)

    # --- LAYER 1: C1 (6 filters, 1 in, 5x5) ---
    c1_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c1.c1.c1.weight.bin", (6 * 1 * 5 * 5,))
    c1_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c1.c1.c1.bias.bin", (6,))
    emit_array("c1_w", c1_w)
    emit_array("c1_b", c1_b)

    # --- LAYER 2: C2 SPLIT ARCHITECTURE ---
    # C2_1 (16 out, 6 in)
    c2_1_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c2_1.c2.c2.weight.bin", (16 * 6 * 5 * 5,))
    c2_1_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c2_1.c2.c2.bias.bin", (16,))
    emit_array("c2_1_w", c2_1_w)
    emit_array("c2_1_b", c2_1_b)

    # C2_2 (16 out, 6 in)
    c2_2_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c2_2.c2.c2.weight.bin", (16 * 6 * 5 * 5,))
    c2_2_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c2_2.c2.c2.bias.bin", (16,))
    emit_array("c2_2_w", c2_2_w)
    emit_array("c2_2_b", c2_2_b)

    # --- LAYER 3: C3 (120 filters, 16 in, 5x5) ---
    c3_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c3.c3.c3.weight.bin", (120 * 16 * 5 * 5,))
    c3_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/c3.c3.c3.bias.bin", (120,))
    emit_array("c3_w", c3_w)
    emit_array("c3_b", c3_b)

    # --- LAYER 4: F4 (84 out, 120 in) ---
    f4_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/f4.f4.f4.weight.bin", (84 * 120,))
    f4_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/f4.f4.f4.bias.bin", (84,))
    emit_array("f4_w", f4_w)
    emit_array("f4_b", f4_b)

    # --- LAYER 5: F5 (10 out, 84 in) ---
    f5_w = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/f5.f5.f5.weight.bin", (10 * 84,))
    f5_b = load_or_rand("/media/omar/983C384F3C382AA0/ara/apps/lenet5_model/model_parameters/f5.f5.f5.bias.bin", (10,))
    emit_array("f5_w", f5_w)
    emit_array("f5_b", f5_b)