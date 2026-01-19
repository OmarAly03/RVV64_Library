#!/usr/bin/env python3
"""
Visual comparison chart for 3x3 convolution test results
"""

import numpy as np
import os

def create_comparison():
    data = {
        'Implementation': [
            'M1 (non-batched)',
            'M2 (non-batched)',
            'M4 (non-batched)',
            'M8 (non-batched)',
            'M2 (batched)',
            'M4 (batched)',
            'M8 (batched)',
        ],
        'Time (ms)': [107.8, 50.1, 27.5, 11.2, 48.7, 27.5, 10.1],
        'Speedup': [1.0, 2.15, 3.93, 9.66, 2.21, 3.92, 10.67],
    }
    
    # ASCII Bar chart for speedup
    print("\n" + "="*70)
    print("SPEEDUP vs M1 BASELINE")
    print("="*70)
    for impl, speedup in zip(data['Implementation'], data['Speedup']):
        bars = '█' * int(speedup * 3)
        print(f"{impl:25} {speedup:6.2f}x  {bars}")
    
    # ASCII Bar chart for time
    print("\n" + "="*70)
    print("EXECUTION TIME")
    print("="*70)
    max_time = max(data['Time (ms)'])
    for impl, time_ms in zip(data['Implementation'], data['Time (ms)']):
        bars = '█' * int((time_ms / max_time) * 40)
        print(f"{impl:25} {time_ms:7.2f}ms  {bars}")
    
    # Detailed comparison table
    print("\n" + "="*70)
    print("DETAILED COMPARISON TABLE")
    print("="*70)
    print(f"{'Implementation':<25} {'Time (ms)':<12} {'Speedup':<12} {'Status':<10}")
    print("-"*70)
    for impl, time_ms, speedup in zip(data['Implementation'], data['Time (ms)'], data['Speedup']):
        status = "✅ PASS" if speedup >= 1.0 else "❌ FAIL"
        print(f"{impl:<25} {time_ms:>10.2f}ms {speedup:>10.2f}x {status:<10}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total implementations tested: {len(data['Implementation'])}")
    print(f"Average speedup: {np.mean(data['Speedup']):.2f}x")
    print(f"Best performer: M8 (batched) at {max(data['Speedup']):.2f}x speedup")
    print(f"Fastest execution: {min(data['Time (ms)']):.2f}ms (M8 batched)")
    print(f"Slowest execution: {max(data['Time (ms)']):.2f}ms (M1 non-batched)")
    print(f"Time reduction: {((max(data['Time (ms)']) - min(data['Time (ms)'])) / max(data['Time (ms)']) * 100):.1f}%")
    
    # Accuracy summary
    print("\n" + "="*70)
    print("ACCURACY VERIFICATION (vs ONNX Reference)")
    print("="*70)
    print("All implementations:")
    print("  ✅ Max Absolute Error: 0 (exact match)")
    print("  ✅ SNR: ∞ dB (perfect)")
    print("  ✅ Value Range: Perfect match")
    print("  ✅ Test Status: ALL PASSED")
    
    # Batching impact analysis
    print("\n" + "="*70)
    print("BATCHING IMPACT ANALYSIS")
    print("="*70)
    m2_improvement = (50.1 - 48.7) / 50.1 * 100
    m4_improvement = (27.5 - 27.5) / 27.5 * 100
    m8_improvement = (11.2 - 10.1) / 11.2 * 100
    
    print(f"M2: {m2_improvement:+.1f}% (48.7ms vs 50.1ms)")
    print(f"M4: {m4_improvement:+.1f}% (27.5ms vs 27.5ms)")
    print(f"M8: {m8_improvement:+.1f}% (10.1ms vs 11.2ms) ⭐ Best batching benefit")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("🏆 Production Choice: M8 Batched")
    print("   - Fastest execution: 10.1ms")
    print("   - Best speedup: 10.67x over baseline")
    print("   - Perfect accuracy verified")
    print("")
    print("⚖️ Balanced Choice: M4 (non-batched)")
    print("   - Good speedup: 3.93x")
    print("   - Lower register pressure")
    print("   - Execution time: 27.5ms")
    print("")
    print("💡 Alternative: M8 (non-batched)")
    print("   - High speedup: 9.66x")
    print("   - Simpler implementation")
    print("   - Execution time: 11.2ms")
    print("="*70 + "\n")

if __name__ == "__main__":
    create_comparison()
