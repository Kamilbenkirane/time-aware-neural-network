"""
Test 03.6: Activation Functions
Tests apply_activation() function for different activation types
Verifies mathematical correctness and edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math

# Import the functions we're testing
from numba_ga import apply_activation, get_activation_name


def run_test():
    print("TEST 03.6: Activation Functions")
    print("=" * 40)
    
    # Test 1: Linear activation (type 0)
    print("Test 1: Linear activation")
    test_values = [-5.0, -1.0, 0.0, 1.0, 5.0, 100.0, -100.0]
    for val in test_values:
        result = apply_activation(val, 0)
        expected = val  # Linear should return input unchanged
        print(f"  linear({val}) = {result}, expected {expected}, match: {abs(result - expected) < 1e-10}")
        assert abs(result - expected) < 1e-10, f"Linear activation failed for {val}"
    
    # Test 2: ReLU activation (type 1)
    print("\nTest 2: ReLU activation")
    test_cases = [
        (-5.0, 0.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (5.0, 5.0),
        (100.0, 100.0)
    ]
    for val, expected in test_cases:
        result = apply_activation(val, 1)
        print(f"  relu({val}) = {result}, expected {expected}, match: {abs(result - expected) < 1e-10}")
        assert abs(result - expected) < 1e-10, f"ReLU activation failed for {val}"
    
    # Test 3: Sigmoid activation (type 2)
    print("\nTest 3: Sigmoid activation")
    test_cases = [
        (0.0, 0.5),
        (1.0, 1.0 / (1.0 + math.exp(-1.0))),
        (-1.0, 1.0 / (1.0 + math.exp(1.0))),
        (10.0, 1.0 / (1.0 + math.exp(-10.0))),
        (-10.0, 1.0 / (1.0 + math.exp(10.0)))
    ]
    for val, expected in test_cases:
        result = apply_activation(val, 2)
        print(f"  sigmoid({val}) = {result:.6f}, expected {expected:.6f}, match: {abs(result - expected) < 1e-10}")
        assert abs(result - expected) < 1e-10, f"Sigmoid activation failed for {val}"
    
    # Test 4: Sigmoid overflow protection
    print("\nTest 4: Sigmoid overflow protection")
    extreme_values = [1000.0, -1000.0, 500.0, -500.0]
    for val in extreme_values:
        result = apply_activation(val, 2)
        print(f"  sigmoid({val}) = {result:.6f}, finite: {math.isfinite(result)}")
        assert math.isfinite(result), f"Sigmoid produced non-finite result for {val}"
        assert 0.0 <= result <= 1.0, f"Sigmoid result {result} outside [0,1] for input {val}"
    
    # Test 5: Tanh activation (type 3)
    print("\nTest 5: Tanh activation")
    test_cases = [
        (0.0, 0.0),
        (1.0, math.tanh(1.0)),
        (-1.0, math.tanh(-1.0)),
        (2.0, math.tanh(2.0)),
        (-2.0, math.tanh(-2.0))
    ]
    for val, expected in test_cases:
        result = apply_activation(val, 3)
        print(f"  tanh({val}) = {result:.6f}, expected {expected:.6f}, match: {abs(result - expected) < 1e-10}")
        assert abs(result - expected) < 1e-10, f"Tanh activation failed for {val}"
    
    # Test 6: Leaky ReLU activation (type 4)
    print("\nTest 6: Leaky ReLU activation")
    test_cases = [
        (-5.0, -5.0 * 0.01),
        (-1.0, -1.0 * 0.01),
        (0.0, 0.0),
        (1.0, 1.0),
        (5.0, 5.0),
        (100.0, 100.0)
    ]
    for val, expected in test_cases:
        result = apply_activation(val, 4)
        print(f"  leaky_relu({val}) = {result}, expected {expected}, match: {abs(result - expected) < 1e-10}")
        assert abs(result - expected) < 1e-10, f"Leaky ReLU activation failed for {val}"
    
    # Test 7: Unknown activation type (should default to linear)
    print("\nTest 7: Unknown activation types")
    unknown_types = [5, 10, -1, 999]
    test_value = 3.14
    for act_type in unknown_types:
        result = apply_activation(test_value, act_type)
        expected = test_value  # Should default to linear
        print(f"  activation_type_{act_type}({test_value}) = {result}, expected {expected} (linear fallback)")
        assert abs(result - expected) < 1e-10, f"Unknown activation type {act_type} should fallback to linear"
    
    # Test 8: Activation name function
    print("\nTest 8: Activation name function")
    name_cases = [
        (0, "linear"),
        (1, "relu"), 
        (2, "sigmoid"),
        (3, "tanh"),
        (4, "leaky_relu"),
        (999, "unknown")
    ]
    for act_type, expected_name in name_cases:
        result_name = get_activation_name(act_type)
        print(f"  get_activation_name({act_type}) = '{result_name}', expected '{expected_name}', match: {result_name == expected_name}")
        assert result_name == expected_name, f"Activation name failed for type {act_type}"
    
    # Test 9: Numerical properties
    print("\nTest 9: Numerical properties")
    
    # Test sigmoid bounds
    for val in np.linspace(-20, 20, 21):
        sigmoid_result = apply_activation(val, 2)
        assert 0.0 <= sigmoid_result <= 1.0, f"Sigmoid out of bounds for {val}: {sigmoid_result}"
    
    # Test tanh bounds
    for val in np.linspace(-20, 20, 21):
        tanh_result = apply_activation(val, 3)
        assert -1.0 <= tanh_result <= 1.0, f"Tanh out of bounds for {val}: {tanh_result}"
    
    print("  ✅ Sigmoid always in [0,1], Tanh always in [-1,1]")
    
    # Test 10: Performance check
    print("\nTest 10: Performance check")
    import time
    
    test_values = np.random.randn(10000)
    
    # Time each activation function
    for activation_type in range(5):
        start_time = time.time()
        for val in test_values:
            _ = apply_activation(float(val), activation_type)
        elapsed = time.time() - start_time
        activation_name = get_activation_name(activation_type)
        print(f"  {activation_name}: {elapsed:.6f} seconds for 10k evaluations")
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   🔢 All activation functions work correctly")
    print(f"   ⚡ Mathematical accuracy verified")
    print(f"   🛡️ Overflow protection working")
    print(f"   🏷️ Name mapping functional")
    print(f"   📊 Numerical bounds respected")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 4: Neural network prediction")
    except Exception as e:
        print(f"\n🛑 TEST 03.6 FAILED: {e}")
        sys.exit(1)