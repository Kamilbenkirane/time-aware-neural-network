"""
Test 03: Population Initialization
Tests initialize_population() function for creating arrays of neural network weights
Tests both basic functionality and parallel compatibility (this is where SIGSEGV might first appear)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numba import njit, prange
import time

# Import the functions we're testing
from numba_ga import initialize_population, initialize_weights, get_total_weights


@njit(fastmath=True, cache=True)
def test_population_sequential(population_sizes, layer_sizes, seeds):
    """Test population initialization sequentially."""
    results = []
    
    for i in range(len(population_sizes)):
        population = initialize_population(population_sizes[i], layer_sizes, seeds[i])
        results.append(population)
    
    return results


@njit(fastmath=True, cache=True, parallel=True)
def test_population_parallel(population_sizes, layer_sizes, seeds):
    """Test population initialization in parallel - POTENTIAL SIGSEGV ZONE."""
    num_tests = len(population_sizes)
    max_pop_size = max(population_sizes)
    total_weights = get_total_weights(layer_sizes)
    
    # Pre-allocate results - 3D array [test_idx, individual_idx, weight_idx]
    results = np.empty((num_tests, max_pop_size, total_weights), dtype=np.float64)
    actual_sizes = np.empty(num_tests, dtype=np.int64)
    
    for i in prange(num_tests):
        population = initialize_population(population_sizes[i], layer_sizes, seeds[i])
        actual_sizes[i] = population.shape[0]
        
        # Copy population to results array
        for j in range(population.shape[0]):
            for k in range(population.shape[1]):
                results[i, j, k] = population[j, k]
    
    return results, actual_sizes


def run_test():
    print("TEST 03: Population Initialization")
    print("=" * 35)
    print("🚨 CRITICAL TEST - This is where SIGSEGV might first occur!")
    print("   Testing 2D array operations and function calls in parallel")
    
    # Test 1: Basic population initialization
    print("\nTest 1: Basic population initialization")
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)
    population_size = 10
    
    population = initialize_population(population_size, layer_sizes, seed=42)
    expected_shape = (population_size, get_total_weights(layer_sizes))
    
    print(f"  Network: 3->5->2")
    print(f"  Population size: {population_size}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Actual shape: {population.shape}")
    print(f"  Shape match: {population.shape == expected_shape}")
    
    assert population.shape == expected_shape, f"Expected shape {expected_shape}, got {population.shape}"
    assert population.dtype == np.float64, f"Expected float64, got {population.dtype}"
    
    # Test 2: Individual diversity within population
    print("\nTest 2: Individual diversity within population")
    layer_sizes = np.array([4, 6, 3], dtype=np.int64)
    population = initialize_population(5, layer_sizes, seed=123)
    
    # Check that individuals are different
    individuals_same = 0
    total_comparisons = 0
    
    for i in range(population.shape[0]):
        for j in range(i+1, population.shape[0]):
            total_comparisons += 1
            if np.array_equal(population[i], population[j]):
                individuals_same += 1
    
    print(f"  Population size: 5")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Identical individuals: {individuals_same}")
    print(f"  All individuals unique: {individuals_same == 0}")
    
    assert individuals_same == 0, f"Found {individuals_same} identical individuals - diversity failed"
    
    # Test 3: Seeded reproducibility
    print("\nTest 3: Seeded reproducibility")
    layer_sizes = np.array([2, 4, 2], dtype=np.int64)
    
    pop1 = initialize_population(8, layer_sizes, seed=999)
    pop2 = initialize_population(8, layer_sizes, seed=999)
    pop3 = initialize_population(8, layer_sizes, seed=111)
    
    print(f"  Same seed populations match: {np.array_equal(pop1, pop2)}")
    print(f"  Different seed populations differ: {not np.array_equal(pop1, pop3)}")
    
    assert np.array_equal(pop1, pop2), "Same seed should produce identical populations"
    assert not np.array_equal(pop1, pop3), "Different seeds should produce different populations"
    
    # Test 4: Sequential vs parallel population creation - CRITICAL TEST
    print("\nTest 4: Sequential vs Parallel Population Creation")
    print("🎯 This is the FIRST potential SIGSEGV test!")
    
    # Test data
    population_sizes = np.array([3, 5, 4], dtype=np.int64)
    layer_sizes = np.array([3, 4, 2], dtype=np.int64)
    seeds = np.array([100, 200, 300], dtype=np.int64)
    
    print(f"  Testing {len(population_sizes)} population initializations")
    print(f"  Population sizes: {population_sizes}")
    print(f"  Network: {'-'.join(map(str, layer_sizes))}")
    
    # Sequential version
    print("\n  4a. Sequential population creation...")
    try:
        start_time = time.time()
        results_seq = test_population_sequential(population_sizes, layer_sizes, seeds)
        seq_time = time.time() - start_time
        print(f"     ✅ Sequential: {seq_time:.6f}s - SUCCESS")
        print(f"     Population shapes: {[pop.shape for pop in results_seq]}")
    except Exception as e:
        print(f"     ❌ Sequential FAILED: {e}")
        return False
    
    # Parallel version - CRITICAL TEST FOR SIGSEGV
    print("\n  4b. Parallel population creation...")
    print("     🚨 Testing 2D array operations in prange loops!")
    print("     🚨 Testing nested function calls in parallel!")
    try:
        start_time = time.time()
        results_par, sizes_par = test_population_parallel(population_sizes, layer_sizes, seeds)
        par_time = time.time() - start_time
        print(f"     ✅ Parallel: {par_time:.6f}s - SUCCESS")
        print(f"     Population sizes: {sizes_par}")
        
        # Performance comparison
        speedup = seq_time / par_time if par_time > 0 else float('inf')
        print(f"     🚀 Speedup: {speedup:.2f}x")
        
        # Check if results match
        matches = True
        for i in range(len(results_seq)):
            seq_pop = results_seq[i]
            par_pop = results_par[i][:sizes_par[i]]  # Extract relevant portion
            
            if not np.allclose(seq_pop, par_pop, rtol=1e-14):
                matches = False
                print(f"     ❌ Population {i} doesn't match!")
                break
        
        if matches:
            print(f"     ✅ All populations match - No race conditions in 2D array operations")
        else:
            print(f"     ❌ Populations don't match - Race condition detected!")
            return False
            
    except Exception as e:
        print(f"     ❌ Parallel CRASHED: {e}")
        print(f"     🚨 SIGSEGV FOUND! Population initialization fails in parallel!")
        print(f"     🔍 This means 2D array operations or function calls are the problem!")
        return False
    
    # Test 5: Different architectures
    print("\nTest 5: Different network architectures")
    
    architectures = [
        [2, 3],           # Simple
        [4, 8, 4],        # Symmetric  
        [1, 1],           # Minimal
        [10, 5, 3, 2],    # Deep
    ]
    
    for arch in architectures:
        layer_sizes = np.array(arch, dtype=np.int64)
        population = initialize_population(6, layer_sizes, seed=1)
        expected_weights = get_total_weights(layer_sizes)
        
        arch_str = "->".join(map(str, arch))
        print(f"  {arch_str}: shape {population.shape}, weights per individual: {expected_weights}")
        assert population.shape == (6, expected_weights), f"Architecture {arch}: wrong shape"
    
    # Test 6: Large population stress test
    print("\nTest 6: Large population stress test")
    layer_sizes = np.array([20, 30, 10], dtype=np.int64)
    large_pop_size = 100
    
    try:
        large_population = initialize_population(large_pop_size, layer_sizes, seed=42)
        expected_shape = (large_pop_size, get_total_weights(layer_sizes))
        print(f"  Large population shape: {large_population.shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Large population SUCCESS: {large_population.shape == expected_shape}")
        assert large_population.shape == expected_shape, "Large population failed"
    except Exception as e:
        print(f"  Large population FAILED: {e}")
        return False
    
    print(f"\n✅ ALL TESTS PASSED - Population initialization works!")
    print(f"   🎉 No SIGSEGV in 2D array operations!")
    print(f"   🎯 Parallel population creation confirmed working!")
    print(f"   📊 Multiple architectures and sizes supported!")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 4: Individual fitness evaluation")
            print(f"   Population operations work in parallel - SIGSEGV must be elsewhere!")
    except Exception as e:
        print(f"\n🛑 TEST 03 FAILED: {e}")
        print(f"   This means the issue is in 2D array operations or function calls in parallel")
        sys.exit(1)