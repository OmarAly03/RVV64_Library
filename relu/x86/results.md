omar@omar-Legion-5:~/Repos/RVV64_Library/relu/x86$ python3 main.py
ONNX model loaded and validated successfully
Model inputs: ['input']
Model outputs: ['output']

ReLU activation: Input tensor size: 1048576 elements
Input range: [-5.000, 5.000]
ONNX reference shape: (1048576,)

Implementation      Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref     0                   inf                 
Python Scalar       0                   inf                 
NumPy maximum       0                   inf                 
C Scalar            0                   inf                 

Verification against ONNX Golden Reference:
NumPy maximum matches ONNX: True
Python scalar matches ONNX: True
C scalar matches ONNX:      True
omar@omar-Legion-5:~/Repos/RVV64_Library/relu/x86$ 