import os
import sys
import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from pyv.kernels import dense  # high-level helper you use in main.py

WEIGHTS_DIR = os.path.join(REPO_ROOT, "models/lenet-5/model_parameters")

def load_bin(filename, shape):
    path = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape).copy()

def main():
    print("=== Testing dense kernel with LeNet-5 weights ===")

    # Load F4 and F5 weights/biases
    f4_w = load_bin("f4.f4.f4.weight.bin", (84, 120))
    f4_b = load_bin("f4.f4.f4.bias.bin",   (84,))
    f5_w = load_bin("f5.f5.f5.weight.bin", (10, 84))
    f5_b = load_bin("f5.f5.f5.bias.bin",   (10,))

    print("f4_w.shape =", f4_w.shape)  # (84, 120)
    print("f4_b.shape =", f4_b.shape)  # (84,)
    print("f5_w.shape =", f5_w.shape)  # (10, 84)
    print("f5_b.shape =", f5_b.shape)  # (10,)

    # Create deterministic inputs
    rng = np.random.default_rng(0)
    f4_in = rng.standard_normal(120, dtype=np.float32)
    f4_in = f4_in.astype(np.float32)

    # ---- Test F4: 120 -> 84 ----
    f4_out = dense(f4_in, f4_w, f4_b, variant="M8")
    print("F4: f4_in.shape  =", f4_in.shape)
    print("F4: f4_out.shape =", f4_out.shape)  # EXPECT (84,)

    # Check shape explicitly
    if f4_out.shape != (84,):
        print("ERROR: F4 dense returned wrong shape, expected (84,), got", f4_out.shape)
    else:
        print("OK: F4 dense shape is (84,)")

    # ---- Test F5: 84 -> 10 ----
    f5_in = f4_out.astype(np.float32)
    f5_out = dense(f5_in, f5_w, f5_b, variant="M8")
    print("F5: f5_in.shape  =", f5_in.shape)
    print("F5: f5_out.shape =", f5_out.shape)  # EXPECT (10,)

    if f5_out.shape != (10,):
        print("ERROR: F5 dense returned wrong shape, expected (10,), got", f5_out.shape)
    else:
        print("OK: F5 dense shape is (10,)")

    # Optional: quick numeric sanity check (should be finite)
    print("F5: min =", float(f5_out.min()), "max =", float(f5_out.max()))

if __name__ == "__main__":
    main()