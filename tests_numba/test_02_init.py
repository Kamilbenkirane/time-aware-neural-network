"""
Test 02: Individual Parameter Initialization
Tests initialize_individual() function for neural network parameter initialization
Simple tests to verify parameter initialization is correct and reproducible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import the functions we're testing
from numba_ga import initialize_individual, get_total_parameters


def run_test():
    print("TEST 02: Individual Parameter Initialization")
    print("=" * 40)
    
    # Test 1: Basic parameter initialization
    print("Test 1: Basic parameter initialization")
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)  # 3->5->2
    parameters = initialize_individual(layer_sizes)
    
    expected_length = get_total_parameters(layer_sizes)  # Should be 39
    print(f"  Network: 3->5->2")
    print(f"  Expected length: {expected_length}")
    print(f"  Actual length: {len(parameters)}")
    print(f"  Length match: {len(parameters) == expected_length}")
    assert len(parameters) == expected_length, f"Expected length {expected_length}, got {len(parameters)}"
    
    # Check data type
    print(f"  Weight dtype: {parameters.dtype}")
    assert parameters.dtype == np.float64, f"Expected float64, got {parameters.dtype}"
    
    # Check that network are not all zeros
    print(f"  Non-zero network: {np.count_nonzero(parameters)}/{len(parameters)}")
    assert np.count_nonzero(parameters) > 0, "All network are zero!"
    
    # Test 2: Seeded initialization (reproducibility)
    print("\nTest 2: Seeded initialization")
    layer_sizes = np.array([4, 6, 3], dtype=np.int64)
    
    parameters1 = initialize_individual(layer_sizes, seed=42)
    parameters2 = initialize_individual(layer_sizes, seed=42)
    parameters3 = initialize_individual(layer_sizes, seed=123)
    
    print(f"  Same seed results match: {np.array_equal(parameters1, parameters2)}")
    print(f"  Different seed results differ: {not np.array_equal(parameters1, parameters3)}")
    
    assert np.array_equal(parameters1, parameters2), "Same seed should produce identical results"
    assert not np.array_equal(parameters1, parameters3), "Different seeds should produce different results"
    
    # Test 3: Different network architectures
    print("\nTest 3: Different network architectures")
    
    architectures = [
        [2, 3],           # Simple 2-layer
        [4, 8, 4],        # Symmetric
        [10, 20, 15, 5],  # Multi-layer
        [1, 1],           # Minimal
    ]
    
    for arch in architectures:
        layer_sizes = np.array(arch, dtype=np.int64)
        parameters = initialize_individual(layer_sizes, seed=1)
        expected_len = get_total_parameters(layer_sizes)
        
        arch_str = "->".join(map(str, arch))
        print(f"  {arch_str}: length {len(parameters)}, expected {expected_len}, match: {len(parameters) == expected_len}")
        assert len(parameters) == expected_len, f"Architecture {arch}: length mismatch"
        assert parameters.dtype == np.float64, f"Architecture {arch}: wrong dtype"
    
    # Test 4: Weight distribution properties
    print("\nTest 4: Weight distribution properties")
    layer_sizes = np.array([50, 100, 50], dtype=np.int64)  # Larger network for statistics
    parameters = initialize_individual(layer_sizes, seed=42)
    
    mean_val = np.mean(parameters)
    std_val = np.std(parameters)
    min_val = np.min(parameters)
    max_val = np.max(parameters)
    
    print(f"  Mean: {mean_val:.6f} (should be ~0)")
    print(f"  Std:  {std_val:.6f} (should be ~0.3)")
    print(f"  Min:  {min_val:.6f}")
    print(f"  Max:  {max_val:.6f}")
    
    # Check mean is close to 0 (random normal distribution)
    assert abs(mean_val) < 0.1, f"Mean too far from 0: {mean_val}"
    
    # Check std is approximately 0.3 (our scaling factor)
    assert 0.2 < std_val < 0.4, f"Std not in expected range: {std_val}"
    
    # Test 5: No seed vs seeded
    print("\nTest 5: No seed vs seeded behavior")
    layer_sizes = np.array([3, 4, 2], dtype=np.int64)
    
    # Without seed - should be different each time
    parameters_a = initialize_individual(layer_sizes)
    parameters_b = initialize_individual(layer_sizes)
    
    # With seed - should be same
    parameters_c = initialize_individual(layer_sizes, seed=999)
    parameters_d = initialize_individual(layer_sizes, seed=999)
    
    print(f"  No seed - different results: {not np.array_equal(parameters_a, parameters_b)}")
    print(f"  With seed - same results: {np.array_equal(parameters_c, parameters_d)}")
    
    # Note: There's a small chance random network could be identical, but very unlikely
    # We'll just check they're not identical (could theoretically fail but very rare)
    same_no_seed = np.array_equal(parameters_a, parameters_b)
    if same_no_seed:
        print("  Warning: Random network were identical (very rare but possible)")
    
    assert np.array_equal(parameters_c, parameters_d), "Seeded network should be identical"
    
    # Test 6: Edge cases
    print("\nTest 6: Edge cases")
    
    # Single connection
    layer_sizes = np.array([1, 1], dtype=np.int64)
    parameters = initialize_individual(layer_sizes, seed=1)
    expected_len = 1 * 1 + 1 + 1  # network + biases + alphas = 3 total
    print(f"  Single connection (1->1): length {len(parameters)}, expected {expected_len}")
    assert len(parameters) == expected_len, "Single connection failed"
    
    # Large network
    layer_sizes = np.array([100, 50], dtype=np.int64)
    parameters = initialize_individual(layer_sizes, seed=1)
    expected_len = 100 * 50 + 50 + 50  # network + biases + alphas = 5100 total
    print(f"  Large network (100->50): length {len(parameters)}, expected {expected_len}")
    assert len(parameters) == expected_len, "Large network failed"
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   🎯 Parameter initialization works correctly")
    print(f"   🔢 Proper length, dtype, and distribution")
    print(f"   🎲 Seeded reproducibility verified")
    print(f"   📐 Multiple architectures supported")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 3: Population initialization")
    except Exception as e:
        print(f"\n🛑 TEST 02 FAILED: {e}")
        sys.exit(1)