#!/usr/bin/env python3
import sys
import numpy as np

# Helper to emit 32-bit integer constants
def emit_val(name, val):
    print(f".global {name}")
    print(f".section .data")
    print(f".balign 4")
    print(f"{name}:")
    print(f"    .word {val}")

# Helper to emit float arrays using raw bytes (Exact float representation)
def emit_array(name, array, alignment='256'):
    print(f".global {name}")
    print(f".section .data")
    print(f".balign {alignment}")
    print(f"{name}:")
    # Convert numpy array to raw bytes
    bs = array.tobytes()
    # Iterate 4 bytes at a time (32-bit words)
    for i in range(0, len(bs), 4):
        # Little Endian handling:
        # array.tobytes() gives bytes in machine order (usually LE for x86/RISC-V)
        # We just print them as a hex word.
        # Construct hex string from 4 bytes reversed for printing as 0x...
        s = ""
        for n in range(4):
            if i+3-n < len(bs):
                s += f"{bs[i+3-n]:02x}"
            else:
                s += "00"
        print(f"    .word 0x{s}")

def maxpool_reference(input_data, C, H, W, K, S):
    out_h = (H - K) // S + 1
    out_w = (W - K) // S + 1
    output = np.zeros((1, C, out_h, out_w), dtype=np.float32)
    
    for c in range(C):
        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * S
                w_start = ow * S
                h_end = min(h_start + K, H)
                w_end = min(w_start + K, W)
                
                slice_hw = input_data[0, c, h_start:h_end, w_start:w_end]
                output[0, c, oh, ow] = np.max(slice_hw)
                
    return output

if __name__ == "__main__":
    # Default to LeNet Pool1: 6ch, 28x28, 2x2, stride 2
    if len(sys.argv) < 6:
        args = [6, 28, 28, 2, 2]
    else:
        args = [int(x) for x in sys.argv[1:6]]
        
    C, H, W, K, S = args
    
    # Generate Data
    np.random.seed(42)
    input_data = np.random.rand(1, C, H, W).astype(np.float32)
    golden = maxpool_reference(input_data, C, H, W, K, S)
    
    # Emit Header
    print(".section .data,\"aw\",@progbits")
    
    # Emit Constants (32-bit matching C++ extern uint32_t)
    emit_val("C_IN", C)
    emit_val("H_IN", H)
    emit_val("W_IN", W)
    emit_val("K_SIZE", K)
    emit_val("STRIDE", S)
    
    # Emit Arrays (Aligned to 256 bytes for best vector load performance)
    emit_array("input_data", input_data, '256')
    emit_array("golden_data", golden, '256')