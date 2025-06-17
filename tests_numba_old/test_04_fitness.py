"""
Test 04: Individual Fitness Evaluation
Tests fitness functions: evaluate_fitness_sphere(), evaluate_fitness_rastrigin(), evaluate_fitness()
Tests individual fitness evaluation with mathematical operations that might cause SIGSEGV in parallel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numba import njit, prange
import time

# Import the functions we're testing
from numba_ga import (
    evaluate_fitness_sphere,
    evaluate_fitness_rastrigin, 
    evaluate_fitness,
    initialize_weights
)


@njit(fastmath=True, cache=True)
def test_fitness_sequential(weights_list, fitness_ids):
    """Test individual fitness evaluation sequentially."""
    num_individuals = len(weights_list)
    fitness_scores = np.empty(num_individuals, dtype=np.float64)
    
    for i in range(num_individuals):
        fitness_scores[i] = evaluate_fitness(weights_list[i], fitness_ids[i])
    
    return fitness_scores


@njit(fastmath=True, cache=True, parallel=True)
def test_fitness_parallel(weights_list, fitness_ids):
    """Test individual fitness evaluation in parallel - POTENTIAL SIGSEGV WITH MATH OPERATIONS."""
    num_individuals = len(weights_list)
    fitness_scores = np.empty(num_individuals, dtype=np.float64)
    
    for i in prange(num_individuals):
        fitness_scores[i] = evaluate_fitness(weights_list[i], fitness_ids[i])
    
    return fitness_scores


def run_test():
    print("TEST 04: Individual Fitness Evaluation")
    print("=" * 38)
    print("🎯 Testing fitness functions with mathematical operations...")
    print("   This tests complex math in parallel that might cause SIGSEGV")
    
    # Test 1: Individual fitness function correctness
    print("\nTest 1: Individual fitness function correctness")
    
    # Create test weights
    test_weights = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    print(f"  Test weights: {test_weights}")
    
    # Test sphere function
    sphere_result = evaluate_fitness_sphere(test_weights)
    expected_sphere = -np.sum(test_weights * test_weights)  # -(1 + 1 + 0.25 + 0.25) = -2.5
    print(f"  Sphere result: {sphere_result:.6f}, expected: {expected_sphere:.6f}")
    assert abs(sphere_result - expected_sphere) < 1e-10, f"Sphere function incorrect: {sphere_result} != {expected_sphere}"
    
    # Test rastrigin function  
    rastrigin_result = evaluate_fitness_rastrigin(test_weights)
    print(f"  Rastrigin result: {rastrigin_result:.6f}")
    # Rastrigin is more complex, just check it's reasonable and different from sphere
    assert abs(rastrigin_result - sphere_result) > 1e-6, "Rastrigin should be different from sphere"
    assert rastrigin_result < 0, "Rastrigin should be negative (for maximization)"
    
    # Test 2: Generic fitness dispatch
    print("\nTest 2: Generic fitness dispatch")
    
    sphere_dispatch = evaluate_fitness(test_weights, 0)
    rastrigin_dispatch = evaluate_fitness(test_weights, 1)
    default_dispatch = evaluate_fitness(test_weights, 999)  # Invalid ID should default to sphere
    
    print(f"  Sphere dispatch: {sphere_dispatch:.6f}")
    print(f"  Rastrigin dispatch: {rastrigin_dispatch:.6f}")  
    print(f"  Default dispatch: {default_dispatch:.6f}")
    
    assert abs(sphere_dispatch - sphere_result) < 1e-14, "Sphere dispatch doesn't match direct call"
    assert abs(rastrigin_dispatch - rastrigin_result) < 1e-14, "Rastrigin dispatch doesn't match direct call"
    assert abs(default_dispatch - sphere_result) < 1e-14, "Default dispatch should use sphere"
    
    # Test 3: Different weight arrays
    print("\nTest 3: Different weight arrays")
    
    layer_sizes = np.array([3, 4, 2], dtype=np.int64)
    test_individuals = []
    
    for i in range(5):
        weights = initialize_weights(layer_sizes, seed=100 + i)
        test_individuals.append(weights)
        
        sphere_fit = evaluate_fitness_sphere(weights)
        rastrigin_fit = evaluate_fitness_rastrigin(weights)
        
        print(f"  Individual {i}: sphere={sphere_fit:.3f}, rastrigin={rastrigin_fit:.3f}")
        
        # Basic sanity checks
        assert np.isfinite(sphere_fit), f"Sphere fitness not finite for individual {i}"
        assert np.isfinite(rastrigin_fit), f"Rastrigin fitness not finite for individual {i}"
        assert sphere_fit < 0, f"Sphere fitness should be negative for individual {i}"
        assert rastrigin_fit < 0, f"Rastrigin fitness should be negative for individual {i}"
    
    # Test 4: Sequential vs parallel fitness evaluation - CRITICAL SIGSEGV TEST
    print("\nTest 4: Sequential vs Parallel Fitness Evaluation")
    print("🚨 This tests mathematical operations in parallel - POTENTIAL SIGSEGV!")
    print("   Testing: complex math (cos, pi, squares) in prange loops")
    
    # Create mixed fitness IDs for testing dispatch in parallel
    fitness_ids = np.array([0, 1, 0, 1, 0], dtype=np.int64)  # Mix of sphere and rastrigin
    print(f"  Testing {len(test_individuals)} individuals")
    print(f"  Fitness IDs: {fitness_ids} (0=sphere, 1=rastrigin)")
    
    # Sequential version
    print("\n  4a. Sequential fitness evaluation...")
    try:
        start_time = time.time()
        fitness_seq = test_fitness_sequential(test_individuals, fitness_ids)
        seq_time = time.time() - start_time
        print(f"     ✅ Sequential: {seq_time:.6f}s - SUCCESS")
        print(f"     Fitness scores: {fitness_seq}")
    except Exception as e:
        print(f"     ❌ Sequential FAILED: {e}")
        return False
    
    # Parallel version - CRITICAL TEST FOR SIGSEGV  
    print("\n  4b. Parallel fitness evaluation...")
    print("     🚨 Testing mathematical operations in prange!")
    print("     🚨 Testing function dispatch in parallel!")
    print("     🚨 Testing cos(), pi, and complex arithmetic!")
    try:
        start_time = time.time()
        fitness_par = test_fitness_parallel(test_individuals, fitness_ids)
        par_time = time.time() - start_time
        print(f"     ✅ Parallel: {par_time:.6f}s - SUCCESS")
        print(f"     Fitness scores: {fitness_par}")
        
        # Performance comparison
        speedup = seq_time / par_time if par_time > 0 else float('inf')
        print(f"     🚀 Speedup: {speedup:.2f}x")
        
        # Check if results match exactly
        if np.allclose(fitness_seq, fitness_par, rtol=1e-14):
            print(f"     ✅ Results match perfectly - No race conditions in math operations")
        else:
            print(f"     ❌ Results don't match - Race condition in mathematical operations!")
            print(f"     Sequential: {fitness_seq}")
            print(f"     Parallel:   {fitness_par}")
            print(f"     Max difference: {np.max(np.abs(fitness_seq - fitness_par))}")
            return False
            
    except Exception as e:
        print(f"     ❌ Parallel CRASHED: {e}")
        print(f"     🚨 SIGSEGV FOUND! Mathematical operations fail in parallel!")
        print(f"     🔍 Issue likely in:")
        print(f"        - Mathematical functions (cos, pi, sqrt)")
        print(f"        - Complex arithmetic operations")
        print(f"        - Function dispatch in parallel context")
        return False
    
    # Test 5: Edge cases and stress testing
    print("\nTest 5: Edge cases and stress testing")
    
    # Very small weights
    tiny_weights = np.array([1e-10, -1e-10, 1e-15], dtype=np.float64)
    sphere_tiny = evaluate_fitness_sphere(tiny_weights)
    rastrigin_tiny = evaluate_fitness_rastrigin(tiny_weights)
    print(f"  Tiny weights - Sphere: {sphere_tiny:.15f}, Rastrigin: {rastrigin_tiny:.6f}")
    assert np.isfinite(sphere_tiny), "Sphere fails with tiny weights"
    assert np.isfinite(rastrigin_tiny), "Rastrigin fails with tiny weights"
    
    # Large weights
    large_weights = np.array([100.0, -50.0, 75.0], dtype=np.float64) 
    sphere_large = evaluate_fitness_sphere(large_weights)
    rastrigin_large = evaluate_fitness_rastrigin(large_weights)
    print(f"  Large weights - Sphere: {sphere_large:.6f}, Rastrigin: {rastrigin_large:.6f}")
    assert np.isfinite(sphere_large), "Sphere fails with large weights"
    assert np.isfinite(rastrigin_large), "Rastrigin fails with large weights"
    
    # Single weight
    single_weight = np.array([2.0], dtype=np.float64)
    sphere_single = evaluate_fitness_sphere(single_weight)
    rastrigin_single = evaluate_fitness_rastrigin(single_weight)
    print(f"  Single weight - Sphere: {sphere_single:.6f}, Rastrigin: {rastrigin_single:.6f}")
    assert np.isfinite(sphere_single), "Sphere fails with single weight"
    assert np.isfinite(rastrigin_single), "Rastrigin fails with single weight"
    
    print(f"\n✅ ALL TESTS PASSED - Individual fitness evaluation works!")
    print(f"   🎉 Mathematical operations work in parallel!")
    print(f"   🎯 Function dispatch works correctly!")
    print(f"   🔢 Both sphere and rastrigin functions operational!")
    print(f"   📊 No SIGSEGV in mathematical computations!")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 5: Population fitness evaluation")
            print(f"   Individual fitness confirmed working - SIGSEGV must be in population-level operations!")
    except Exception as e:
        print(f"\n🛑 TEST 04 FAILED: {e}")
        print(f"   This means the issue is in mathematical operations or function dispatch")
        sys.exit(1)