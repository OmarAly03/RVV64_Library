#!/usr/bin/env python3
"""
Main test script for Conv2D implementations.
Tests both scalar NumPy implementation and ONNX model execution.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sys
import os

# Add the src directory to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from conv2d_numpy import conv2d_scalar, conv2d_scalar_flattened
from onnx_utils import snr_db, max_abs_error

def test_numpy_scalar_implementation():
    """Test the NumPy scalar convolution implementation."""
    print("=" * 60)
    print("Testing NumPy Scalar Conv2D Implementation")
    print("=" * 60)
    
    # Test case 1: Simple 4x4 input with 3x3 kernel
    print("\n1. Testing 4x4 input with 3x3 kernel:")
    in_h, in_w, in_c = 4, 4, 1
    k_h, k_w, out_c = 3, 3, 1
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    
    # Input: 4x4x1, sequential values
    input_arr = np.arange(1, 17, dtype=np.float32).reshape((in_h, in_w, in_c))
    print(f"Input shape: {input_arr.shape}")
    print(f"Input:\n{input_arr[:,:,0]}")
    
    # Kernel: 3x3, edge detection-like
    kernel = np.array([[[[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]]]], dtype=np.float32)
    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel:\n{kernel[0,0,:,:]}")
    
    # Perform convolution
    output = conv2d_scalar(input_arr, kernel, stride_h, stride_w, pad_h, pad_w)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output[:,:,0]}")
    
    # Test case 2: Multi-channel input
    print("\n2. Testing multi-channel convolution:")
    in_h, in_w, in_c = 5, 5, 3  # RGB image
    k_h, k_w, out_c = 3, 3, 2   # 2 output channels
    
    # Random input
    np.random.seed(42)
    input_arr = np.random.randn(in_h, in_w, in_c).astype(np.float32)
    kernel = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32) * 0.1
    
    output = conv2d_scalar(input_arr, kernel, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    print(f"Input shape: {input_arr.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output channel 0 sample:\n{output[:3,:3,0]}")
    
    # Test case 3: Flattened version
    print("\n3. Testing flattened implementation:")
    input_flat = input_arr.flatten()
    kernel_flat = kernel.flatten()
    output_flat = conv2d_scalar_flattened(
        input_flat, in_h, in_w, in_c, kernel_flat, k_h, k_w, out_c,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1
    )
    
    # Compare flattened vs normal implementation
    diff = np.max(np.abs(output - output_flat))
    print(f"Max difference between normal and flattened: {diff}")
    if diff < 1e-6:
        print("✅ Flattened implementation matches normal implementation")
    else:
        print("❌ Flattened implementation differs from normal implementation")
    
    return True

def create_and_test_onnx_model():
    """Create ONNX model and test it."""
    print("\n" + "=" * 60)
    print("Creating and Testing ONNX Conv2D Model")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = "./output_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a smaller ONNX model for testing
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None, None, None])
    
    # Smaller weight tensor for testing: 4 output channels, 3 input channels, 3x3 kernel
    weight_shape = [4, 3, 3, 3]
    np.random.seed(42)
    weight_data = np.random.randn(*weight_shape).astype(np.float32) * 0.1
    weight_tensor = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=weight_shape,
        vals=weight_data.flatten().tolist()
    )
    
    # Bias tensor
    bias_shape = [4]
    bias_data = np.random.randn(*bias_shape).astype(np.float32) * 0.01
    bias_tensor = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=bias_shape,
        vals=bias_data.flatten().tolist()
    )
    
    # Create Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[conv_node],
        name="TestConvolutionGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor, bias_tensor]
    )
    
    # Create model
    model = helper.make_model(
        graph, 
        producer_name="conv_test", 
        opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 7
    
    # Save model
    model_path = os.path.join(output_dir, "test_convolution.onnx")
    onnx.save(model, model_path)
    print(f"✅ ONNX model saved to: {model_path}")
    
    # Test the ONNX model
    print("\nTesting ONNX model execution:")
    
    try:
        # Load and run ONNX model
        ort_session = ort.InferenceSession(model_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, 8, 8).astype(np.float32)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {ort_output.shape}")
        print(f"Output statistics:")
        print(f"  Mean: {np.mean(ort_output):.6f}")
        print(f"  Std:  {np.std(ort_output):.6f}")
        print(f"  Min:  {np.min(ort_output):.6f}")
        print(f"  Max:  {np.max(ort_output):.6f}")
        
        print("✅ ONNX model executed successfully")
        return True, test_input, ort_output, weight_data, bias_data
        
    except Exception as e:
        print(f"❌ Error running ONNX model: {e}")
        return False, None, None, None, None

def compare_numpy_vs_onnx():
    """Compare NumPy implementation with ONNX model output."""
    print("\n" + "=" * 60)
    print("Comparing NumPy vs ONNX Implementation")
    print("=" * 60)
    
    # Create and test ONNX model
    success, test_input, onnx_output, weight_data, bias_data = create_and_test_onnx_model()
    
    if not success:
        print("❌ Cannot compare - ONNX model failed")
        return
    
    # Convert ONNX input format (NCHW) to NumPy format (HWC)
    # test_input shape: (1, 3, 8, 8) -> (8, 8, 3)
    numpy_input = test_input[0].transpose(1, 2, 0)  # CHW -> HWC
    
    # Convert ONNX kernel format (OIHW) to NumPy format (OIHW is already correct)
    numpy_kernel = weight_data  # Shape: (4, 3, 3, 3)
    
    print(f"Converted input shape: {numpy_input.shape}")
    print(f"Kernel shape: {numpy_kernel.shape}")
    
    # Run NumPy convolution
    numpy_output = conv2d_scalar(numpy_input, numpy_kernel, 
                                stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    
    # Add bias to NumPy output
    numpy_output_with_bias = numpy_output + bias_data.reshape(1, 1, -1)
    
    # Convert NumPy output (HWC) to ONNX format (NCHW)
    numpy_output_nchw = numpy_output_with_bias.transpose(2, 0, 1)  # HWC -> CHW
    numpy_output_nchw = numpy_output_nchw[np.newaxis, :]  # Add batch dimension
    
    print(f"NumPy output shape: {numpy_output_nchw.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    
    # Compare outputs
    if numpy_output_nchw.shape == onnx_output.shape:
        max_error = max_abs_error(onnx_output, numpy_output_nchw)
        snr = snr_db(onnx_output, numpy_output_nchw)
        
        print(f"\nComparison Results:")
        print(f"Max absolute error: {max_error:.8f}")
        print(f"SNR (dB): {snr:.2f}")
        
        if max_error < 1e-5:
            print("✅ NumPy and ONNX implementations match closely!")
        elif max_error < 1e-3:
            print("⚠️  NumPy and ONNX implementations have small differences")
        else:
            print("❌ NumPy and ONNX implementations differ significantly")
            
        # Show sample comparison
        print(f"\nSample values comparison (first channel, top-left 3x3):")
        print("ONNX output:")
        print(onnx_output[0, 0, :3, :3])
        print("NumPy output:")
        print(numpy_output_nchw[0, 0, :3, :3])
        
    else:
        print("❌ Output shapes don't match - cannot compare")

def run_performance_test():
    """Run a simple performance comparison."""
    print("\n" + "=" * 60)
    print("Performance Test")
    print("=" * 60)
    
    import time
    
    # Test parameters
    in_h, in_w, in_c = 32, 32, 3
    k_h, k_w, out_c = 3, 3, 16
    
    # Generate random data
    np.random.seed(42)
    input_arr = np.random.randn(in_h, in_w, in_c).astype(np.float32)
    kernel = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32) * 0.1
    
    print(f"Input: {input_arr.shape}, Kernel: {kernel.shape}")
    
    # Time NumPy scalar implementation
    start_time = time.time()
    for _ in range(10):  # Run 10 times for better timing
        output = conv2d_scalar(input_arr, kernel, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    numpy_time = (time.time() - start_time) / 10
    
    print(f"NumPy scalar implementation: {numpy_time*1000:.2f} ms per run")
    print(f"Output shape: {output.shape}")

def main():
    """Main test function."""
    print("Conv2D Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: NumPy scalar implementation
        test_numpy_scalar_implementation()
        
        # Test 2: ONNX model creation and execution
        create_and_test_onnx_model()
        
        # Test 3: Compare implementations
        compare_numpy_vs_onnx()
        
        # Test 4: Performance test
        run_performance_test()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install numpy onnx onnxruntime")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()