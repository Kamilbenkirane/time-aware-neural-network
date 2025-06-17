"""
Test neural network prediction functions with temporal patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from numba_ga import (
    initialize_parameters, 
    initialize_population,
    predict_individual,
    predict_population, 
    predict_individual_stateless,
    reset_population_memory,
    compute_layer_indices
)

def test_basic_prediction():
    """Test basic individual prediction functionality."""
    print("=== Test: Basic Individual Prediction ===")
    
    # Simple network: 2 -> 3 -> 1
    layer_sizes = np.array([2, 3, 1], dtype=np.int64)
    activations = np.array([1, 2], dtype=np.int64)  # ReLU -> Sigmoid
    
    # Initialize parameters
    parameters = initialize_parameters(layer_sizes, seed=42)
    
    # Test input
    current_time = 10.0
    x_vector = np.array([0.5, -0.3], dtype=np.float64)
    inputs = (current_time, x_vector)
    
    # Initialize memory
    total_neurons = sum(layer_sizes[1:])  # 3 + 1 = 4
    prev_states = np.zeros(total_neurons, dtype=np.float64)
    prev_time = 0.0
    
    # Pre-compute indices
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)
    
    # Predict
    outputs, new_states, timestamp = predict_individual(
        parameters, layer_sizes, activations, inputs, 
        prev_states, prev_time, param_indices, neuron_indices
    )
    
    print(f"Input: {x_vector}")
    print(f"Output: {outputs}")
    print(f"State shape: {new_states.shape}")
    print(f"Timestamp: {timestamp}")
    
    # Verify shapes
    assert outputs.shape == (1,), f"Expected output shape (1,), got {outputs.shape}"
    assert new_states.shape == (4,), f"Expected state shape (4,), got {new_states.shape}"
    assert timestamp == current_time
    
    print("✓ Basic prediction test passed\n")


def test_population_prediction():
    """Test population batch prediction."""
    print("=== Test: Population Prediction ===")
    
    layer_sizes = np.array([1, 2, 1], dtype=np.int64)
    activations = np.array([1, 0], dtype=np.int64)  # ReLU -> Linear
    pop_size = 5
    
    # Initialize population
    population = initialize_population(pop_size, layer_sizes, seed=123)
    
    # Initialize memory states
    population_states, population_times = reset_population_memory(layer_sizes, pop_size)
    
    # Test input
    current_time = 5.0
    x_vector = np.array([1.0], dtype=np.float64)
    inputs = (current_time, x_vector)
    
    # Predict
    outputs, updated_states, updated_times = predict_population(
        population, layer_sizes, activations, inputs,
        population_states, population_times
    )
    
    print(f"Population size: {pop_size}")
    print(f"Output shape: {outputs.shape}")
    print(f"Sample outputs: {outputs[:3, 0]}")
    
    # Verify shapes
    assert outputs.shape == (pop_size, 1), f"Expected output shape ({pop_size}, 1), got {outputs.shape}"
    assert updated_states.shape == population_states.shape
    assert np.all(updated_times == current_time)
    
    print("✓ Population prediction test passed\n")


def test_temporal_patterns():
    """Test neural network response to temporal input patterns."""
    print("=== Test: Temporal Pattern Response ===")
    
    # Define temporal patterns
    patterns = {
        'Sparse': '1000100010001',      # Sparse pulses
        'Alternating': '1010101010101',  # Regular alternating 
        'Burst': '1111111000000000'      # Burst then silence
    }
    
    # Simple network for pattern testing
    layer_sizes = np.array([1, 4, 1], dtype=np.int64)
    activations = np.array([3, 0], dtype=np.int64)  # Tanh -> Linear
    
    # Initialize individual
    parameters = initialize_parameters(layer_sizes, seed=999)
    
    # Pre-compute indices
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)
    
    for pattern_name, pattern_str in patterns.items():
        print(f"\n--- Pattern: {pattern_name} ({pattern_str}) ---")
        
        # Reset memory
        total_neurons = sum(layer_sizes[1:])
        prev_states = np.zeros(total_neurons, dtype=np.float64)
        prev_time = 0.0
        
        outputs = []
        states_history = []
        
        # Process pattern over time
        for i, bit in enumerate(pattern_str):
            current_time = float(i * 5)  # 5ms intervals
            input_value = float(bit)
            x_vector = np.array([input_value], dtype=np.float64)
            inputs = (current_time, x_vector)
            
            # Predict
            output, new_states, _ = predict_individual(
                parameters, layer_sizes, activations, inputs,
                prev_states, prev_time, param_indices, neuron_indices
            )
            
            outputs.append(output[0])
            states_history.append(new_states.copy())
            
            # Update for next iteration
            prev_states = new_states
            prev_time = current_time
        
        outputs = np.array(outputs)
        
        print(f"Input pattern:  {pattern_str}")
        print(f"Output pattern: {['%.3f' % x for x in outputs]}")
        print(f"Output range:   [{outputs.min():.3f}, {outputs.max():.3f}]")
        print(f"Output std:     {outputs.std():.3f}")
        
        # Verify temporal dependency (different patterns should produce different outputs)
        assert len(outputs) == len(pattern_str)
        assert not np.allclose(outputs, outputs[0])  # Should vary over time
    
    print("\n✓ Temporal pattern test passed\n")


def test_memory_persistence():
    """Test that neural network memory persists correctly between predictions."""
    print("=== Test: Memory Persistence ===")
    
    layer_sizes = np.array([1, 3, 1], dtype=np.int64)
    activations = np.array([1, 2], dtype=np.int64)  # ReLU -> Sigmoid
    
    parameters = initialize_parameters(layer_sizes, seed=777)
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)
    
    # Test scenario: same input at different times
    x_vector = np.array([1.0], dtype=np.float64)
    total_neurons = sum(layer_sizes[1:])
    
    # Prediction 1: Fresh start
    prev_states = np.zeros(total_neurons, dtype=np.float64)
    prev_time = 0.0
    current_time = 0.0
    inputs = (current_time, x_vector)
    
    output1, states1, _ = predict_individual(
        parameters, layer_sizes, activations, inputs,
        prev_states, prev_time, param_indices, neuron_indices
    )
    
    # Prediction 2: Immediate follow-up (no time gap)
    current_time = 0.0  # Same time
    inputs = (current_time, x_vector)
    
    output2, states2, _ = predict_individual(
        parameters, layer_sizes, activations, inputs,
        states1, prev_time, param_indices, neuron_indices
    )
    
    # Prediction 3: After long time gap (memory should completely decay)
    current_time = 100.0  # Long time gap → complete memory decay
    inputs = (current_time, x_vector)
    
    output3, states3, _ = predict_individual(
        parameters, layer_sizes, activations, inputs,
        states1, 0.0, param_indices, neuron_indices
    )
    
    # Prediction 4: After moderate time gap (partial decay)
    current_time = 5.0  # Moderate time gap → partial decay
    inputs = (current_time, x_vector)
    
    output4, states4, _ = predict_individual(
        parameters, layer_sizes, activations, inputs,
        states1, 0.0, param_indices, neuron_indices
    )
    
    print(f"Same input at different times:")
    print(f"Output 1 (t=0, fresh):     {output1[0]:.6f}")
    print(f"Output 2 (t=0, immediate): {output2[0]:.6f}") 
    print(f"Output 3 (t=100, complete decay): {output3[0]:.6f}")
    print(f"Output 4 (t=5, partial decay):    {output4[0]:.6f}")
    
    # Verify temporal behavior
    assert not np.isclose(output1[0], output2[0]), "Memory should affect immediate prediction"
    assert np.isclose(output1[0], output3[0], rtol=1e-5), "Complete decay should match fresh start"
    assert not np.isclose(output1[0], output4[0]), "Partial decay should differ from fresh start"
    assert not np.isclose(output2[0], output4[0]), "Partial decay should differ from immediate"
    
    print("✓ Memory persistence test passed\n")


def test_stateless_compatibility():
    """Test that stateless prediction matches memory-based prediction with zero states."""
    print("=== Test: Stateless Compatibility ===")
    
    layer_sizes = np.array([2, 3, 2], dtype=np.int64)
    activations = np.array([2, 1], dtype=np.int64)  # Sigmoid -> ReLU
    
    parameters = initialize_parameters(layer_sizes, seed=555)
    x_vector = np.array([0.3, -0.7], dtype=np.float64)
    
    # Stateless prediction
    output_stateless = predict_individual_stateless(
        parameters, layer_sizes, activations, x_vector
    )
    
    # Memory-based prediction with zero initial states
    current_time = 0.0
    inputs = (current_time, x_vector)
    total_neurons = sum(layer_sizes[1:])
    prev_states = np.zeros(total_neurons, dtype=np.float64)
    prev_time = 0.0
    
    param_indices, neuron_indices = compute_layer_indices(layer_sizes)
    
    output_memory, _, _ = predict_individual(
        parameters, layer_sizes, activations, inputs,
        prev_states, prev_time, param_indices, neuron_indices
    )
    
    print(f"Stateless output: {output_stateless}")
    print(f"Memory output:    {output_memory}")
    print(f"Difference:       {np.abs(output_stateless - output_memory)}")
    
    # Should be identical for zero initial states
    assert np.allclose(output_stateless, output_memory, rtol=1e-10), \
        "Stateless and memory-based predictions should match with zero states"
    
    print("✓ Stateless compatibility test passed\n")


def test_performance_benchmark():
    """Benchmark prediction performance."""
    print("=== Test: Performance Benchmark ===")
    
    layer_sizes = np.array([10, 20, 10, 1], dtype=np.int64)  # Larger network
    activations = np.array([1, 2, 3], dtype=np.int64)  # ReLU -> Sigmoid -> Tanh
    pop_size = 1000
    
    # Initialize
    population = initialize_population(pop_size, layer_sizes, seed=111)
    population_states, population_times = reset_population_memory(layer_sizes, pop_size)
    
    x_vector = np.random.randn(10).astype(np.float64)
    current_time = 0.0
    inputs = (current_time, x_vector)
    
    # Warm-up (JIT compilation)
    _ = predict_population(
        population[:10], layer_sizes, activations, inputs,
        population_states[:10], population_times[:10]
    )
    
    # Benchmark
    start_time = time.time()
    outputs, _, _ = predict_population(
        population, layer_sizes, activations, inputs,
        population_states, population_times
    )
    end_time = time.time()
    
    duration = (end_time - start_time) * 1000  # Convert to milliseconds
    predictions_per_ms = pop_size / duration
    
    print(f"Population size: {pop_size}")
    print(f"Network size: {layer_sizes}")
    print(f"Prediction time: {duration:.2f} ms")
    print(f"Throughput: {predictions_per_ms:.1f} predictions/ms")
    print(f"Output shape: {outputs.shape}")
    
    # Performance assertion (should be reasonably fast)
    assert duration < 100, f"Prediction too slow: {duration:.2f} ms"
    
    print("✓ Performance benchmark passed\n")


def run_all_tests():
    """Run all prediction tests."""
    print("🧠 NEURAL NETWORK PREDICTION TESTS")
    print("=" * 50)
    
    test_basic_prediction()
    test_population_prediction()
    test_temporal_patterns()
    test_memory_persistence()
    test_stateless_compatibility()
    test_performance_benchmark()
    
    print("🎉 All prediction tests passed!")


if __name__ == "__main__":
    run_all_tests()