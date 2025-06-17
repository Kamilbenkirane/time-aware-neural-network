"""
Test 05: Population Fitness Evaluation
Simple test for evaluate_population_fitness() with parallel=True
🚨 CRITICAL TEST - This is where our original SIGSEGV most likely occurred!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

# Import the functions we're testing
from numba_ga import (
    evaluate_population_fitness,
    initialize_population
)


def run_test():
    print("TEST 05: Population Fitness Evaluation")
    print("=" * 40)
    print("🚨 CRITICAL TEST - Testing evaluate_population_fitness() with parallel=True")
    print("   This is exactly where our original SIGSEGV occurred!")
    
    # Test 1: Basic population fitness evaluation
    print("\nTest 1: Basic population fitness evaluation")
    
    layer_sizes = np.array([4, 6, 3], dtype=np.int64)
    population_size = 10
    population = initialize_population(population_size, layer_sizes, seed=42)
    
    print(f"  Population shape: {population.shape}")
    print(f"  Testing both sphere (id=0) and rastrigin (id=1) fitness")
    
    # Test sphere fitness
    try:
        start_time = time.time()
        fitness_sphere = evaluate_population_fitness(population, 0)
        sphere_time = time.time() - start_time
        
        print(f"  ✅ Sphere fitness: {sphere_time:.6f}s - SUCCESS")
        print(f"  Sphere fitness shape: {fitness_sphere.shape}")
        print(f"  Sphere fitness range: {np.min(fitness_sphere):.3f} to {np.max(fitness_sphere):.3f}")
        
        assert len(fitness_sphere) == population_size, f"Wrong fitness array length: {len(fitness_sphere)} != {population_size}"
        assert fitness_sphere.dtype == np.float64, f"Wrong fitness dtype: {fitness_sphere.dtype}"
        assert np.all(np.isfinite(fitness_sphere)), "Some fitness values are not finite"
        
    except Exception as e:
        print(f"  ❌ Sphere fitness FAILED: {e}")
        return False
    
    # Test rastrigin fitness
    try:
        start_time = time.time()
        fitness_rastrigin = evaluate_population_fitness(population, 1)
        rastrigin_time = time.time() - start_time
        
        print(f"  ✅ Rastrigin fitness: {rastrigin_time:.6f}s - SUCCESS")
        print(f"  Rastrigin fitness shape: {fitness_rastrigin.shape}")
        print(f"  Rastrigin fitness range: {np.min(fitness_rastrigin):.3f} to {np.max(fitness_rastrigin):.3f}")
        
        # Should be different fitness functions
        assert not np.array_equal(fitness_sphere, fitness_rastrigin), "Different fitness functions should give different results"
        
    except Exception as e:
        print(f"  ❌ Rastrigin fitness FAILED: {e}")
        return False
    
    # Test 2: Different population sizes
    print("\nTest 2: Different population sizes")
    
    test_sizes = [5, 20, 50]
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)
    
    for size in test_sizes:
        print(f"  Testing population size: {size}")
        
        try:
            population = initialize_population(size, layer_sizes, seed=100 + size)
            start_time = time.time()
            fitness = evaluate_population_fitness(population, 0)
            eval_time = time.time() - start_time
            
            print(f"    ✅ Size {size}: {eval_time:.6f}s - SUCCESS")
            print(f"    Fitness range: {np.min(fitness):.3f} to {np.max(fitness):.3f}")
            
            assert len(fitness) == size, f"Wrong result length for size {size}"
            assert np.all(np.isfinite(fitness)), f"Non-finite values for size {size}"
            
        except Exception as e:
            print(f"    ❌ Size {size} FAILED: {e}")
            return False
    
    # Test 3: Different neural network architectures
    print("\nTest 3: Different neural network architectures")
    
    architectures = [
        [2, 3],      # Simple 2-layer
        [3, 5, 2],   # 3-layer
        [4, 6, 4, 2] # 4-layer
    ]
    
    for arch in architectures:
        arch_str = "->".join(map(str, arch))
        print(f"  Testing architecture: {arch_str}")
        
        try:
            layer_sizes = np.array(arch, dtype=np.int64)
            population = initialize_population(8, layer_sizes, seed=200)
            
            start_time = time.time()
            fitness = evaluate_population_fitness(population, 1)  # Use rastrigin
            eval_time = time.time() - start_time
            
            print(f"    ✅ Architecture {arch_str}: {eval_time:.6f}s - SUCCESS")
            print(f"    Population shape: {population.shape}")
            print(f"    Fitness range: {np.min(fitness):.3f} to {np.max(fitness):.3f}")
            
            assert len(fitness) == 8, f"Wrong result length for architecture {arch}"
            assert np.all(np.isfinite(fitness)), f"Non-finite values for architecture {arch}"
            
        except Exception as e:
            print(f"    ❌ Architecture {arch_str} FAILED: {e}")
            return False
    
    # Test 4: Large population stress test
    print("\nTest 4: Large population stress test")
    
    layer_sizes = np.array([20, 30, 10], dtype=np.int64)
    large_pop_size = 100
    
    try:
        large_population = initialize_population(large_pop_size, layer_sizes, seed=999)
        print(f"  Large population: {large_population.shape}")
        
        start_time = time.time()
        large_fitness = evaluate_population_fitness(large_population, 0)
        eval_time = time.time() - start_time
        
        print(f"  ✅ Large population: {eval_time:.6f}s - SUCCESS")
        print(f"  Fitness range: {np.min(large_fitness):.3f} to {np.max(large_fitness):.3f}")
        print(f"  Results shape: {large_fitness.shape}")
        
        assert len(large_fitness) == large_pop_size, "Large population size mismatch"
        assert np.all(np.isfinite(large_fitness)), "Large population non-finite results"
        
    except Exception as e:
        print(f"  ❌ Large population FAILED: {e}")
        return False
    
    print(f"\n✅ ALL TESTS PASSED - Population fitness evaluation works with parallel=True!")
    print(f"   🎉 No SIGSEGV in basic population fitness evaluation!")
    print(f"   🎯 evaluate_population_fitness() confirmed working!")
    print(f"   🔍 Ready to test more complex genetic operations!")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 6: Tournament selection")
            print(f"   Population fitness confirmed working - SIGSEGV must be in selection/genetic operations!")
    except Exception as e:
        print(f"\n🛑 TEST 05 FAILED: {e}")
        print(f"   This means the issue is in basic population-level fitness evaluation")
        sys.exit(1)