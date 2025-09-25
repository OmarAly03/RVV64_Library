```bash
omar@omar-Legion-5:~/riscv-dev/onnx_project/x86$ python3 main.py
ONNX model loaded and validated successfully
Model inputs: ['a', 'b']
Model outputs: ['c']

Matrix multiplication: A(512x512) @ B(512x512) -> C(512x512)
Total operations: 268,435,456 FLOPs
ONNX reference shape: (512, 512)

Implementation      Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref     0                   inf                 
Python Scalar       0.000198364         129.892             
NumPy dot           0.00012207          135.354             
NumPy @ operator    0.00012207          135.354             
C Scalar            0.000198364         129.892             

Verification against ONNX Golden Reference:
NumPy @ matches ONNX:     True
NumPy dot matches ONNX:   True
Python scalar matches ONNX: True
C scalar matches ONNX:    True
omar@omar-Legion-5:~/riscv-dev/onnx_project/x86$ 

```