"""
Test 01: Weight Calculation Functions
Tests get_layer_weights() and get_total_weights() functions
Simple tests to verify weight calculations are correct
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import the functions we're testing
from numba_ga import get_layer_weights, get_total_weights


def run_test():
    print("TEST 01: Weight Calculation Functions")
    print("=" * 35)
    
    # Test 1: Single layer connection
    print("Test 1: Single layer connection")
    layer_sizes = np.array([4, 6], dtype=np.int64)  # 4 -> 6
    layer_weights = get_layer_weights(layer_sizes, 0)
    expected = 4 * 6 + 6  # weights + biases = 24 + 6 = 30
    print(f"  4->6: got {layer_weights}, expected {expected}, match: {layer_weights == expected}")
    assert layer_weights == expected, f"Expected {expected}, got {layer_weights}"
    
    # Test 2: Different layer sizes
    print("\nTest 2: Different layer sizes")
    test_cases = [
        ([2, 3], 0, 2*3 + 3),     # 2->3: 6+3=9
        ([5, 1], 0, 5*1 + 1),     # 5->1: 5+1=6
        ([10, 20], 0, 10*20 + 20), # 10->20: 200+20=220
    ]
    
    for layers, layer_id, expected in test_cases:
        layer_sizes = np.array(layers, dtype=np.int64)
        result = get_layer_weights(layer_sizes, layer_id)
        print(f"  {layers[0]}->{layers[1]}: got {result}, expected {expected}, match: {result == expected}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    # Test 3: Total weights for simple networks
    print("\nTest 3: Total weights calculation")
    
    # Simple 2-layer network: 3->5->2
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)
    total = get_total_weights(layer_sizes)
    # Layer 0->1: 3*5 + 5 = 20
    # Layer 1->2: 5*2 + 2 = 12
    # Total: 20 + 12 = 32
    expected = (3*5 + 5) + (5*2 + 2)
    print(f"  3->5->2: got {total}, expected {expected}, match: {total == expected}")
    assert total == expected, f"Expected {expected}, got {total}"
    
    # Test 4: Multi-layer network
    print("\nTest 4: Multi-layer network")
    layer_sizes = np.array([4, 8, 6, 3, 2], dtype=np.int64)  # 4->8->6->3->2
    total = get_total_weights(layer_sizes)
    # Layer 0->1: 4*8 + 8 = 40
    # Layer 1->2: 8*6 + 6 = 54  
    # Layer 2->3: 6*3 + 3 = 21
    # Layer 3->4: 3*2 + 2 = 8
    # Total: 40 + 54 + 21 + 8 = 123
    expected = (4*8 + 8) + (8*6 + 6) + (6*3 + 3) + (3*2 + 2)
    print(f"  4->8->6->3->2: got {total}, expected {expected}, match: {total == expected}")
    assert total == expected, f"Expected {expected}, got {total}"
    
    # Test 5: Individual layer verification for multi-layer
    print("\nTest 5: Individual layer verification")
    layer_sizes = np.array([4, 8, 6, 3, 2], dtype=np.int64)
    
    layer_results = []
    for layer_id in range(len(layer_sizes) - 1):
        weights = get_layer_weights(layer_sizes, layer_id)
        layer_results.append(weights)
        from_size = layer_sizes[layer_id]
        to_size = layer_sizes[layer_id + 1]
        expected = from_size * to_size + to_size
        print(f"  Layer {layer_id} ({from_size}->{to_size}): got {weights}, expected {expected}")
        assert weights == expected, f"Layer {layer_id}: expected {expected}, got {weights}"
    
    # Verify sum matches total
    sum_of_layers = sum(layer_results)
    total_direct = get_total_weights(layer_sizes)
    print(f"  Sum of individual layers: {sum_of_layers}")
    print(f"  Total from get_total_weights: {total_direct}")
    print(f"  Match: {sum_of_layers == total_direct}")
    assert sum_of_layers == total_direct, f"Sum mismatch: {sum_of_layers} != {total_direct}"
    
    # Test 6: Edge cases
    print("\nTest 6: Edge cases")
    
    # Single connection (2 layers)
    layer_sizes = np.array([1, 1], dtype=np.int64)
    total = get_total_weights(layer_sizes)
    expected = 1*1 + 1  # 2
    print(f"  1->1: got {total}, expected {expected}, match: {total == expected}")
    assert total == expected, f"Expected {expected}, got {total}"
    
    # Large layer
    layer_sizes = np.array([100, 50], dtype=np.int64)
    total = get_total_weights(layer_sizes)
    expected = 100*50 + 50  # 5050
    print(f"  100->50: got {total}, expected {expected}, match: {total == expected}")
    assert total == expected, f"Expected {expected}, got {total}"
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   🎯 Weight calculation functions work correctly")
    print(f"   📊 Verified: single layers, multi-layers, edge cases")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 2: Individual weight initialization")
    except Exception as e:
        print(f"\n🛑 TEST 01 FAILED: {e}")
        sys.exit(1)