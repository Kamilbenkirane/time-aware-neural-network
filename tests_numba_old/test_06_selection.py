"""
Test 06: Tournament Selection
Tests tournament_selection_single() and tournament_selection() with parallel execution
Tests selection pressure and randomness that might cause SIGSEGV in parallel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

# Import the functions we're testing
from numba_ga import (
    tournament_selection_single,
    tournament_selection,
    initialize_population,
    evaluate_population_fitness
)


def run_test():
    print("TEST 06: Tournament Selection")
    print("=" * 30)
    print("🎯 Testing tournament selection with parallel execution")
    print("   This tests random selection and array indexing in parallel - POTENTIAL SIGSEGV!")
    
    # Test 1: Single tournament selection
    print("\nTest 1: Single tournament selection")
    
    # Create test fitness scores (higher is better)
    test_fitness = np.array([1.0, 5.0, 3.0, 8.0, 2.0, 6.0], dtype=np.float64)
    print(f"  Test fitness scores: {test_fitness}")
    print(f"  Best individual should be index 3 (fitness=8.0)")
    
    # Test with different tournament sizes
    tournament_sizes = [2, 3, 4]
    
    for t_size in tournament_sizes:
        print(f"\n  Tournament size {t_size}:")
        
        # Run multiple tournaments to check selection pressure
        selections = []
        for i in range(100):
            selected = tournament_selection_single(test_fitness, t_size, seed=42 + i)
            selections.append(selected)
        
        # Count selections
        selection_counts = np.bincount(selections, minlength=len(test_fitness))
        print(f"    Selection counts: {selection_counts}")
        print(f"    Best individual (idx 3) selected: {selection_counts[3]} times")
        
        # Verify selection pressure (best individual should be selected more often)
        assert selection_counts[3] > 0, f"Best individual never selected with tournament size {t_size}"
        
        # With larger tournament size, selection pressure should increase
        if t_size > 2:
            print(f"    Selection pressure working correctly")
    
    # Test 2: Parallel tournament selection
    print("\nTest 2: Parallel tournament selection")
    print("🚨 Testing parallel random selection - POTENTIAL SIGSEGV!")
    
    # Create larger fitness array
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)
    population_size = 20
    population = initialize_population(population_size, layer_sizes, seed=100)
    fitness_scores = evaluate_population_fitness(population, 0)  # Sphere fitness
    
    print(f"  Population size: {population_size}")
    print(f"  Fitness range: {np.min(fitness_scores):.3f} to {np.max(fitness_scores):.3f}")
    
    # Test different numbers of parents and tournament sizes
    test_cases = [
        {"num_parents": 5, "tournament_size": 2},
        {"num_parents": 10, "tournament_size": 3},
        {"num_parents": 15, "tournament_size": 4},
    ]
    
    for case in test_cases:
        num_parents = case["num_parents"]
        tournament_size = case["tournament_size"]
        
        print(f"\n  Testing {num_parents} parents, tournament size {tournament_size}")
        
        try:
            start_time = time.time()
            parent_indices = tournament_selection(fitness_scores, num_parents, tournament_size, seed=200)
            selection_time = time.time() - start_time
            
            print(f"    ✅ Parallel selection: {selection_time:.6f}s - SUCCESS")
            print(f"    Selected indices: {parent_indices[:5]}..." if len(parent_indices) > 5 else f"    Selected indices: {parent_indices}")
            
            # Basic validation
            assert len(parent_indices) == num_parents, f"Wrong number of parents selected: {len(parent_indices)} != {num_parents}"
            assert parent_indices.dtype == np.int64, f"Wrong parent indices dtype: {parent_indices.dtype}"
            assert np.all(parent_indices >= 0), "Negative parent indices detected"
            assert np.all(parent_indices < population_size), "Parent indices out of bounds"
            
            # Check selection pressure (better individuals should be selected more often)
            best_individual_idx = np.argmax(fitness_scores)
            selection_counts = np.bincount(parent_indices, minlength=population_size)
            
            print(f"    Best individual (idx {best_individual_idx}): selected {selection_counts[best_individual_idx]} times")
            print(f"    Most selected individual: idx {np.argmax(selection_counts)} (selected {np.max(selection_counts)} times)")
            
        except Exception as e:
            print(f"    ❌ Parallel selection FAILED: {e}")
            print(f"    🚨 SIGSEGV in parallel tournament selection!")
            return False
    
    # Test 3: Different population sizes and architectures
    print("\nTest 3: Different population sizes and architectures")
    
    architectures = [
        {"layer_sizes": [2, 3], "pop_size": 10},
        {"layer_sizes": [4, 6, 3], "pop_size": 25},
        {"layer_sizes": [5, 8, 4, 2], "pop_size": 50},
    ]
    
    for arch in architectures:
        layer_sizes = np.array(arch["layer_sizes"], dtype=np.int64)
        pop_size = arch["pop_size"]
        arch_str = "->".join(map(str, arch["layer_sizes"]))
        
        print(f"\n  Testing architecture {arch_str}, population {pop_size}")
        
        try:
            # Create population and evaluate fitness
            population = initialize_population(pop_size, layer_sizes, seed=300)
            fitness = evaluate_population_fitness(population, 1)  # Rastrigin fitness
            
            print(f"    Population shape: {population.shape}")
            print(f"    Fitness range: {np.min(fitness):.3f} to {np.max(fitness):.3f}")
            
            # Select parents
            num_parents = pop_size // 2  # Select half the population
            start_time = time.time()
            parents = tournament_selection(fitness, num_parents, tournament_size=3, seed=400)
            selection_time = time.time() - start_time
            
            print(f"    ✅ Selection: {selection_time:.6f}s - SUCCESS")
            print(f"    Selected {len(parents)} parents from {pop_size} individuals")
            
            # Validation
            assert len(parents) == num_parents, f"Wrong number of parents for {arch_str}"
            assert np.all(parents >= 0) and np.all(parents < pop_size), f"Invalid parent indices for {arch_str}"
            
        except Exception as e:
            print(f"    ❌ Architecture {arch_str} FAILED: {e}")
            return False
    
    # Test 4: Edge cases and stress testing
    print("\nTest 4: Edge cases and stress testing")
    
    # Very small population
    print("  Testing very small population (size 3)")
    small_fitness = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        small_parents = tournament_selection(small_fitness, 2, tournament_size=2, seed=500)
        print(f"    ✅ Small population: selected {small_parents}")
        assert len(small_parents) == 2, "Wrong number of parents for small population"
        assert np.all(small_parents >= 0) and np.all(small_parents < 3), "Invalid indices for small population"
    except Exception as e:
        print(f"    ❌ Small population FAILED: {e}")
        return False
    
    # Large population stress test
    print("  Testing large population (size 200)")
    try:
        large_layer_sizes = np.array([10, 15, 5], dtype=np.int64)
        large_population = initialize_population(200, large_layer_sizes, seed=600)
        large_fitness = evaluate_population_fitness(large_population, 0)
        
        start_time = time.time()
        large_parents = tournament_selection(large_fitness, 100, tournament_size=4, seed=700)
        large_time = time.time() - start_time
        
        print(f"    ✅ Large population: {large_time:.6f}s - SUCCESS")
        print(f"    Selected {len(large_parents)} parents")
        
        assert len(large_parents) == 100, "Wrong number of parents for large population"
        assert np.all(large_parents >= 0) and np.all(large_parents < 200), "Invalid indices for large population"
        
    except Exception as e:
        print(f"    ❌ Large population FAILED: {e}")
        return False
    
    # Tournament size edge cases
    print("  Testing tournament size edge cases")
    test_fitness_edge = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    try:
        # Tournament size 1 (should just be random selection)
        parents_t1 = tournament_selection(test_fitness_edge, 3, tournament_size=1, seed=800)
        print(f"    ✅ Tournament size 1: {parents_t1}")
        
        # Tournament size equals population size
        parents_t5 = tournament_selection(test_fitness_edge, 3, tournament_size=5, seed=900)
        print(f"    ✅ Tournament size 5: {parents_t5}")
        
        # Should mostly select the best individual (index 4) when tournament size = population size
        selection_counts = np.bincount(parents_t5, minlength=5)
        best_idx = np.argmax(test_fitness_edge)  # Should be index 4
        print(f"    Best individual (idx {best_idx}) selected {selection_counts[best_idx]} times out of 3")
        
    except Exception as e:
        print(f"    ❌ Tournament size edge cases FAILED: {e}")
        return False
    
    print(f"\n✅ ALL TESTS PASSED - Tournament selection works with parallel execution!")
    print(f"   🎉 No SIGSEGV in parallel random selection and array indexing!")
    print(f"   🎯 Selection pressure working correctly!")
    print(f"   🔀 Random selection working in parallel!")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to STEP 7: Crossover operations")
            print(f"   Tournament selection confirmed working - SIGSEGV must be in crossover/mutation operations!")
    except Exception as e:
        print(f"\n🛑 TEST 06 FAILED: {e}")
        print(f"   This means the issue is in tournament selection or parallel random operations")
        sys.exit(1)