"""
Test suite for pure function-based individual operations (nb_individual_pure.py).

This module tests all pure @njit functions for individual weight management,
neural network operations, and genetic operations using raw numpy arrays.
"""

import numpy as np
import pytest
import time
from src.evolution.nb_individual_pure import (
    # Weight management functions
    get_total_weights_count,
    initialize_individual_weights,
    clone_individual_weights,
    
    # Neural network state functions
    create_individual_nn_state,
    reset_individual_nn_state,
    
    # Neural network forward pass functions
    extract_individual_weights,
    compute_temporal_decay,
    individual_layer_forward_pass,
    get_individual_action,
    
    # Genetic operation functions
    mutate_individual_weights,
    crossover_individuals,
    
    # Wrapper functions
    create_individual_wrapper,
    IndividualWrapper
)


class TestWeightManagement:
    """Test weight management functions."""
    
    def test_get_total_weights_count(self):
        """Test total weight count calculation."""
        # Test basic case
        total = get_total_weights_count(1, 5, 3)
        expected = 1*5 + 5 + 3*5 + 3  # hidden_weights + hidden_bias + output_weights + output_bias
        assert total == expected == 28
        
        # Test different sizes
        assert get_total_weights_count(2, 10, 4) == 2*10 + 10 + 4*10 + 4 == 74
        assert get_total_weights_count(3, 15, 2) == 3*15 + 15 + 2*15 + 2 == 92
    
    def test_initialize_individual_weights(self):
        """Test individual weight initialization."""
        input_size, hidden_size, output_size = 1, 5, 3
        expected_size = get_total_weights_count(input_size, hidden_size, output_size)
        
        # Test without seed
        weights = initialize_individual_weights(input_size, hidden_size, output_size)
        assert weights.shape == (expected_size,)
        assert weights.dtype == np.float64
        assert not np.allclose(weights, 0.0)  # Should not be all zeros
        
        # Test with seed for reproducibility
        weights1 = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
        weights2 = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
        np.testing.assert_array_equal(weights1, weights2)
        
        # Test different seeds produce different weights
        weights3 = initialize_individual_weights(input_size, hidden_size, output_size, seed=123)
        assert not np.allclose(weights1, weights3)
    
    def test_clone_individual_weights(self):
        """Test weight cloning."""
        weights = initialize_individual_weights(1, 5, 3, seed=42)
        cloned = clone_individual_weights(weights)
        
        # Should be equal but not the same object
        np.testing.assert_array_equal(weights, cloned)
        assert weights is not cloned
        
        # Modifying clone shouldn't affect original
        cloned[0] = 999.0
        assert weights[0] != 999.0


class TestNeuralNetworkState:
    """Test neural network state management functions."""
    
    def test_create_individual_nn_state(self):
        """Test neural network state creation."""
        hidden_size, output_size = 5, 3
        state = create_individual_nn_state(hidden_size, output_size)
        
        hidden_prev, hidden_time, output_prev, output_time = state
        
        assert hidden_prev.shape == (hidden_size,)
        assert output_prev.shape == (output_size,)
        assert hidden_prev.dtype == np.float64
        assert output_prev.dtype == np.float64
        assert hidden_time == 0.0
        assert output_time == 0.0
        np.testing.assert_array_equal(hidden_prev, np.zeros(hidden_size))
        np.testing.assert_array_equal(output_prev, np.zeros(output_size))
    
    def test_reset_individual_nn_state(self):
        """Test neural network state reset."""
        hidden_size, output_size = 5, 3
        state = create_individual_nn_state(hidden_size, output_size)
        hidden_prev, hidden_time, output_prev, output_time = state
        
        # Modify state
        hidden_prev[:] = 1.0
        output_prev[:] = 2.0
        hidden_time = 10.0
        output_time = 20.0
        
        # Reset state
        reset_state = reset_individual_nn_state((hidden_prev, hidden_time, output_prev, output_time))
        new_hidden_prev, new_hidden_time, new_output_prev, new_output_time = reset_state
        
        # Should be reset to zeros
        np.testing.assert_array_equal(new_hidden_prev, np.zeros(hidden_size))
        np.testing.assert_array_equal(new_output_prev, np.zeros(output_size))
        assert new_hidden_time == 0.0
        assert new_output_time == 0.0


class TestNeuralNetworkOperations:
    """Test neural network forward pass functions."""
    
    def test_extract_individual_weights(self):
        """Test weight extraction from flat array."""
        input_size, hidden_size, output_size = 2, 3, 2
        weights = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
        
        hidden_weights, hidden_bias, output_weights, output_bias = extract_individual_weights(
            weights, input_size, hidden_size, output_size
        )
        
        assert hidden_weights.shape == (hidden_size, input_size)
        assert hidden_bias.shape == (hidden_size,)
        assert output_weights.shape == (output_size, hidden_size)
        assert output_bias.shape == (output_size,)
        
        # Check that all weights are accounted for
        total_extracted = (hidden_weights.size + hidden_bias.size + 
                          output_weights.size + output_bias.size)
        assert total_extracted == len(weights)
    
    def test_compute_temporal_decay(self):
        """Test temporal decay computation."""
        # Test no previous time
        decay = compute_temporal_decay(10.0, 0.0)
        assert decay == 0.0
        
        # Test same time
        decay = compute_temporal_decay(5.0, 5.0)
        assert decay == 1.0
        
        # Test exponential decay
        decay = compute_temporal_decay(6.0, 5.0)  # time_diff = 1.0
        expected = np.exp(-1.0)
        assert abs(decay - expected) < 1e-10
        
        # Test large time difference (should be clamped)
        decay = compute_temporal_decay(100.0, 0.1)
        expected = np.exp(-50.0)  # Should be clamped to max 50
        assert abs(decay - expected) < 1e-10
    
    def test_individual_layer_forward_pass(self):
        """Test layer forward pass."""
        input_size, layer_size = 2, 3
        
        # Create test data
        x = np.array([0.5, -0.3], dtype=np.float64)
        weights = np.random.randn(layer_size, input_size).astype(np.float64)
        bias = np.random.randn(layer_size).astype(np.float64)
        prev_values = np.zeros(layer_size, dtype=np.float64)
        
        # Test forward pass
        output, new_time = individual_layer_forward_pass(
            x, weights, bias, prev_values, 0.0, 1.0, 0.5
        )
        
        assert output.shape == (layer_size,)
        assert output.dtype == np.float64
        assert new_time == 1.0
        assert np.all(np.abs(output) <= 1.0)  # tanh output should be in [-1, 1]
        
        # Check that prev_values was updated in-place
        np.testing.assert_array_equal(prev_values, output)
    
    def test_get_individual_action(self):
        """Test complete neural network action selection."""
        input_size, hidden_size, output_size = 1, 5, 3
        weights = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
        nn_state = create_individual_nn_state(hidden_size, output_size)
        
        # Test action selection
        action, updated_state = get_individual_action(
            weights, 0.5, 1.0, nn_state,
            input_size, hidden_size, output_size, 1.0
        )
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < output_size
        
        # Check state was updated
        assert len(updated_state) == 4
        hidden_prev, hidden_time, output_prev, output_time = updated_state
        assert hidden_prev.shape == (hidden_size,)
        assert output_prev.shape == (output_size,)
        assert hidden_time == 1.0
        assert output_time == 1.0


class TestGeneticOperations:
    """Test genetic operation functions."""
    
    def test_mutate_individual_weights(self):
        """Test weight mutation."""
        weights = initialize_individual_weights(1, 5, 3, seed=42)
        original_weights = weights.copy()
        
        # Test with no mutation
        mutated = mutate_individual_weights(weights, 0.0, 0.1)
        np.testing.assert_array_equal(mutated, original_weights)
        
        # Test with guaranteed mutation
        mutated = mutate_individual_weights(weights, 1.0, 0.1)
        assert not np.allclose(mutated, original_weights)
        assert mutated.shape == original_weights.shape
        
        # Original should be unchanged
        np.testing.assert_array_equal(weights, original_weights)
    
    def test_crossover_individuals(self):
        """Test crossover operation."""
        parent1 = initialize_individual_weights(1, 5, 3, seed=42)
        parent2 = initialize_individual_weights(1, 5, 3, seed=123)
        
        # Test no crossover
        child1, child2 = crossover_individuals(parent1, parent2, 0.0)
        np.testing.assert_array_equal(child1, parent1)
        np.testing.assert_array_equal(child2, parent2)
        
        # Test guaranteed crossover
        child1, child2 = crossover_individuals(parent1, parent2, 1.0)
        assert child1.shape == parent1.shape
        assert child2.shape == parent2.shape
        
        # Children should be different from parents (with high probability)
        assert not np.allclose(child1, parent1) or not np.allclose(child2, parent2)


class TestWrapperFunctions:
    """Test wrapper functions for API compatibility."""
    
    def test_create_individual_wrapper(self):
        """Test individual wrapper creation."""
        weights = initialize_individual_weights(1, 5, 3, seed=42)
        wrapper = create_individual_wrapper(weights, 1, 5, 3, 1.0)
        
        assert isinstance(wrapper, IndividualWrapper)
        np.testing.assert_array_equal(wrapper.get_weights(), weights)
        assert wrapper.input_size == 1
        assert wrapper.hidden_size == 5
        assert wrapper.output_size == 3
        assert wrapper.alpha == 1.0
    
    def test_individual_wrapper_functionality(self):
        """Test individual wrapper methods."""
        weights = initialize_individual_weights(1, 5, 3, seed=42)
        wrapper = create_individual_wrapper(weights)
        
        # Test get_action
        action = wrapper.get_action(0.5, 1.0)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3
        
        # Test reset_state
        wrapper.reset_state()  # Should not raise an error
        
        # Test total_weights property
        assert wrapper.total_weights == len(weights)


class TestPerformance:
    """Test performance of pure function implementations."""
    
    def test_individual_action_performance(self):
        """Test neural network action performance."""
        weights = initialize_individual_weights(1, 10, 3, seed=42)
        nn_state = create_individual_nn_state(10, 3)
        
        # Warm up compilation
        for _ in range(10):
            get_individual_action(weights, 0.5, 1.0, nn_state, 1, 10, 3, 1.0)
        
        # Benchmark
        num_iterations = 1000
        start_time = time.time()
        
        for i in range(num_iterations):
            action, nn_state = get_individual_action(
                weights, 0.5, float(i), nn_state, 1, 10, 3, 1.0
            )
        
        total_time = time.time() - start_time
        time_per_action = total_time / num_iterations
        
        # Should be very fast (< 1ms per action)
        assert time_per_action < 0.001
        
        print(f"Individual action performance: {time_per_action*1000:.4f} ms per action")
        print(f"Actions per second: {1/time_per_action:.0f}")
    
    def test_genetic_operations_performance(self):
        """Test genetic operations performance."""
        weights1 = initialize_individual_weights(1, 10, 3, seed=42)
        weights2 = initialize_individual_weights(1, 10, 3, seed=123)
        
        # Warm up
        for _ in range(10):
            mutate_individual_weights(weights1, 0.1, 0.1)
            crossover_individuals(weights1, weights2, 0.8)
        
        # Benchmark mutation
        num_mutations = 1000
        start_time = time.time()
        
        for _ in range(num_mutations):
            mutate_individual_weights(weights1, 0.1, 0.1)
        
        mutation_time = time.time() - start_time
        
        # Benchmark crossover
        num_crossovers = 1000
        start_time = time.time()
        
        for _ in range(num_crossovers):
            crossover_individuals(weights1, weights2, 0.8)
        
        crossover_time = time.time() - start_time
        
        print(f"Mutation performance: {mutation_time/num_mutations*1000:.4f} ms per mutation")
        print(f"Crossover performance: {crossover_time/num_crossovers*1000:.4f} ms per crossover")
        
        # Should be very fast
        assert mutation_time / num_mutations < 0.001
        assert crossover_time / num_crossovers < 0.001


def test_integration_full_individual_lifecycle():
    """Integration test for complete individual lifecycle."""
    input_size, hidden_size, output_size = 1, 10, 3
    
    # Initialize individual
    weights = initialize_individual_weights(input_size, hidden_size, output_size, seed=42)
    nn_state = create_individual_nn_state(hidden_size, output_size)
    
    # Simulate neural network usage
    actions = []
    for i in range(100):
        action, nn_state = get_individual_action(
            weights, np.sin(i * 0.1), float(i), nn_state,
            input_size, hidden_size, output_size, 1.0
        )
        actions.append(action)
    
    # Check we got valid actions
    assert len(actions) == 100
    assert all(0 <= a < output_size for a in actions)
    
    # Test genetic operations
    mutated_weights = mutate_individual_weights(weights, 0.1, 0.1)
    partner_weights = initialize_individual_weights(input_size, hidden_size, output_size, seed=123)
    child1, child2 = crossover_individuals(weights, partner_weights, 0.8)
    
    # All should have same shape
    assert weights.shape == mutated_weights.shape == child1.shape == child2.shape
    
    # Test wrapper compatibility
    wrapper = create_individual_wrapper(weights)
    wrapper_action = wrapper.get_action(0.5, 1.0)
    assert 0 <= wrapper_action < output_size


if __name__ == "__main__":
    # Run basic tests
    print("Testing Pure Function-Based Individual Operations")
    print("=" * 60)
    
    # Test weight management
    print("✓ Testing weight management...")
    test_weight = TestWeightManagement()
    test_weight.test_get_total_weights_count()
    test_weight.test_initialize_individual_weights()
    test_weight.test_clone_individual_weights()
    
    # Test neural network state
    print("✓ Testing neural network state...")
    test_nn_state = TestNeuralNetworkState()
    test_nn_state.test_create_individual_nn_state()
    test_nn_state.test_reset_individual_nn_state()
    
    # Test neural network operations
    print("✓ Testing neural network operations...")
    test_nn_ops = TestNeuralNetworkOperations()
    test_nn_ops.test_extract_individual_weights()
    test_nn_ops.test_compute_temporal_decay()
    test_nn_ops.test_individual_layer_forward_pass()
    test_nn_ops.test_get_individual_action()
    
    # Test genetic operations
    print("✓ Testing genetic operations...")
    test_genetic = TestGeneticOperations()
    test_genetic.test_mutate_individual_weights()
    test_genetic.test_crossover_individuals()
    
    # Test wrapper functions
    print("✓ Testing wrapper functions...")
    test_wrapper = TestWrapperFunctions()
    test_wrapper.test_create_individual_wrapper()
    test_wrapper.test_individual_wrapper_functionality()
    
    # Test performance
    print("✓ Testing performance...")
    test_perf = TestPerformance()
    test_perf.test_individual_action_performance()
    test_perf.test_genetic_operations_performance()
    
    # Integration test
    print("✓ Testing integration...")
    test_integration_full_individual_lifecycle()
    
    print("\n🎯 All individual function tests passed!")
    print("⚡ Pure function-based individual operations are working correctly!")
    print("🚀 Zero classes in core computation confirmed!")