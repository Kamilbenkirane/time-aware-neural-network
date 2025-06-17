"""
Test 01: Time-Aware Network Parameter Management
Tests get_layer_parameters() and get_total_parameters() for time-aware networks with per-neuron alpha
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import the functions we're testing
from numba_ga import (
    get_layer_parameters,
    get_total_parameters
)


def run_test():
    print("TEST 01: Time-Aware Network Parameter Management")
    print("=" * 50)
    print("🎯 Testing parameter counting for time-aware networks with per-neuron alpha")
    
    # Test 1: Basic time-aware network architecture
    print("\nTest 1: Basic time-aware network (1->10->3)")
    
    layer_sizes = np.array([1, 10, 3], dtype=np.int64)  # input_size=1, hidden_size=10, output_size=3
    print(f"  Architecture: {layer_sizes}")
    
    # Test layer parameter counting
    hidden_params = get_layer_parameters(layer_sizes, 0)  # layer 0->1
    output_params = get_layer_parameters(layer_sizes, 1)  # layer 1->2
    
    print(f"  Hidden layer parameters (1->10): {hidden_params}")
    print(f"    Expected: {1*10} weights + {10} biases + {10} alphas = {1*10 + 10 + 10}")
    expected_hidden = 1*10 + 10 + 10  # weights + biases + alphas
    assert hidden_params == expected_hidden, f"Hidden params wrong: {hidden_params} != {expected_hidden}"
    
    print(f"  Output layer parameters (10->3): {output_params}")
    print(f"    Expected: {10*3} weights + {3} biases + {3} alphas = {10*3 + 3 + 3}")
    expected_output = 10*3 + 3 + 3  # weights + biases + alphas
    assert output_params == expected_output, f"Output params wrong: {output_params} != {expected_output}"
    
    # Test total parameter counting
    total_params = get_total_parameters(layer_sizes)
    expected_total = hidden_params + output_params
    
    print(f"  Total parameters: {total_params}")
    print(f"    Expected: {hidden_params} + {output_params} = {expected_total}")
    assert total_params == expected_total, f"Total params wrong: {total_params} != {expected_total}"
    
    print(f"  ✅ Basic architecture test passed!")
    
    # Test 2: Different architectures
    print("\nTest 2: Different architectures")
    
    test_architectures = [
        [2, 5, 2],      # Small network (3 layers)
        [4, 8, 3],      # Medium network (3 layers)
        [10, 20, 5],    # Larger network (3 layers)
        [1, 1, 1],      # Minimal network (3 layers)
        [3, 12, 6],     # Custom network (3 layers)
        [2, 4, 6, 3],   # 4-layer network
        [5, 8, 10, 6, 2], # 5-layer network
        [1, 2, 3, 4, 1], # 5-layer varying sizes
        [10, 15, 20, 15, 10, 5], # 6-layer network
    ]
    
    for arch in test_architectures:
        layer_sizes = np.array(arch, dtype=np.int64)
        arch_str = "->".join(map(str, arch))
        
        print(f"\n  Testing architecture: {arch_str}")
        
        # Calculate expected values manually for any number of layers
        expected_total = 0
        layer_details = []
        
        for layer_id in range(len(arch) - 1):
            from_size = arch[layer_id]
            to_size = arch[layer_id + 1]
            layer_params = from_size * to_size + to_size + to_size  # weights + biases + alphas
            expected_total += layer_params
            layer_details.append(f"Layer {layer_id}->({from_size}->{to_size}): {layer_params} params")
        
        # Test our functions
        actual_total = get_total_parameters(layer_sizes)
        
        print(f"    Total layers: {len(arch)} ({len(arch)-1} connections)")
        for detail in layer_details:
            print(f"      {detail}")
        print(f"    Total: {actual_total} (expected {expected_total})")
        
        # Test individual layer parameter calculations
        for layer_id in range(len(arch) - 1):
            from_size = arch[layer_id]
            to_size = arch[layer_id + 1]
            expected_layer = from_size * to_size + to_size + to_size
            actual_layer = get_layer_parameters(layer_sizes, layer_id)
            assert actual_layer == expected_layer, f"Layer {layer_id} mismatch for {arch_str}: {actual_layer} != {expected_layer}"
        
        assert actual_total == expected_total, f"Total mismatch for {arch_str}: {actual_total} != {expected_total}"
        
        print(f"    ✅ {arch_str} passed!")
    
    # Test 3: Edge cases
    print("\nTest 3: Edge cases and verification")
    
    # Very large architecture
    large_arch = [100, 200, 50]
    layer_sizes = np.array(large_arch, dtype=np.int64)
    total_params = get_total_parameters(layer_sizes)
    
    # Manual calculation
    hidden_calc = 100*200 + 200 + 200  # 20000 + 200 + 200 = 20400
    output_calc = 200*50 + 50 + 50     # 10000 + 50 + 50 = 10100
    expected_large = hidden_calc + output_calc  # 20400 + 10100 = 30500
    
    print(f"  Large architecture {large_arch}: {total_params} parameters")
    print(f"    Hidden: {hidden_calc}, Output: {output_calc}, Total expected: {expected_large}")
    assert total_params == expected_large, f"Large architecture failed: {total_params} != {expected_large}"
    print(f"    ✅ Large architecture passed!")
    
    # Single hidden neuron
    tiny_arch = [1, 1, 1]
    layer_sizes = np.array(tiny_arch, dtype=np.int64)
    total_params = get_total_parameters(layer_sizes)
    
    # Manual calculation
    hidden_calc = 1*1 + 1 + 1  # 1 + 1 + 1 = 3
    output_calc = 1*1 + 1 + 1  # 1 + 1 + 1 = 3
    expected_tiny = hidden_calc + output_calc  # 3 + 3 = 6
    
    print(f"  Tiny architecture {tiny_arch}: {total_params} parameters")
    print(f"    Hidden: {hidden_calc}, Output: {output_calc}, Total expected: {expected_tiny}")
    assert total_params == expected_tiny, f"Tiny architecture failed: {total_params} != {expected_tiny}"
    print(f"    ✅ Tiny architecture passed!")
    
    # Test 4: Deep network verification
    print("\nTest 4: Deep network (multi-layer) verification")
    
    # Test a specific deep network in detail
    deep_arch = [3, 8, 12, 6, 2]  # 5-layer network with 4 connections
    layer_sizes = np.array(deep_arch, dtype=np.int64)
    
    print(f"  Deep architecture {deep_arch} (5 layers, 4 connections):")
    
    total_calculated = 0
    for i in range(len(deep_arch) - 1):
        from_size = deep_arch[i]
        to_size = deep_arch[i + 1]
        layer_params = get_layer_parameters(layer_sizes, i)
        expected = from_size * to_size + to_size + to_size
        
        print(f"    Connection {i}: {from_size}->{to_size}")
        print(f"      Weights: {from_size}x{to_size} = {from_size * to_size}")
        print(f"      Biases: {to_size}")
        print(f"      Alphas: {to_size}")
        print(f"      Total: {layer_params} (expected {expected})")
        
        assert layer_params == expected, f"Deep network layer {i} failed"
        total_calculated += layer_params
    
    total_params = get_total_parameters(layer_sizes)
    print(f"    Deep network total: {total_params} (calculated sum: {total_calculated})")
    assert total_params == total_calculated, "Deep network total mismatch"
    print(f"    ✅ Deep network verification passed!")
    
    # Test 5: Verify per-neuron alpha concept
    print("\nTest 5: Per-neuron alpha verification")
    
    arch = [2, 4, 3]
    layer_sizes = np.array(arch, dtype=np.int64)
    
    hidden_params = get_layer_parameters(layer_sizes, 0)
    output_params = get_layer_parameters(layer_sizes, 1)
    
    print(f"  Architecture {arch}:")
    print(f"    Hidden layer (2->4): {hidden_params} parameters")
    print(f"      - 8 weights (2x4)")
    print(f"      - 4 biases (one per hidden neuron)")
    print(f"      - 4 alphas (one per hidden neuron)")
    print(f"    Output layer (4->3): {output_params} parameters")
    print(f"      - 12 weights (4x3)")
    print(f"      - 3 biases (one per output neuron)")
    print(f"      - 3 alphas (one per output neuron)")
    
    # Verify the math
    assert hidden_params == 8 + 4 + 4, "Hidden layer calculation wrong"
    assert output_params == 12 + 3 + 3, "Output layer calculation wrong"
    print(f"    ✅ Per-neuron alpha structure verified!")
    
    print(f"\n✅ ALL TESTS PASSED - Parameter management for time-aware networks works!")
    print(f"   🎯 Layer parameter counting correct (weights + biases + alphas)!")
    print(f"   🧠 Per-neuron alpha parameters confirmed!")
    print(f"   📊 Architecture flexibility validated!")
    print(f"   🔢 Math verification complete!")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 2: Individual parameter initialization")
            print(f"   Parameter counting confirmed working for time-aware networks with per-neuron alpha!")
    except Exception as e:
        print(f"\n🛑 TEST 01 FAILED: {e}")
        print(f"   This means the issue is in parameter counting functions")
        sys.exit(1)