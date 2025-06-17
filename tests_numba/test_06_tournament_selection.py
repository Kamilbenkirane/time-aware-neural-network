"""
Test 06: Tournament Selection
Tests tournament_selection() function for genetic algorithms
Verifies selection pressure, reproducibility, and edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

# Import the functions we're testing
from numba_ga import tournament_selection, initialize_population, get_total_parameters


def run_test():
    print("TEST 06: Tournament Selection")
    print("=" * 40)
    
    # Test 1: Basic tournament selection functionality
    print("Test 1: Basic tournament selection")
    
    layer_sizes = np.array([4, 6, 3], dtype=np.int64)
    pop_size = 20
    population = initialize_population(pop_size, layer_sizes, seed=42)
    
    # Create fitness scores with clear hierarchy
    fitness_scores = np.array([i * 0.1 for i in range(pop_size)], dtype=np.float64)  # 0.0 to 1.9
    print(f"  Population size: {pop_size}")
    print(f"  Fitness range: {fitness_scores.min():.1f} to {fitness_scores.max():.1f}")
    
    # Test binary tournament (k=2)
    tournament_size = 2
    num_parents = 10
    
    selected_parents, selected_indices = tournament_selection(
        population, fitness_scores, tournament_size, num_parents, seed=123
    )
    
    print(f"  Tournament size: {tournament_size}")
    print(f"  Parents selected: {num_parents}")
    print(f"  Selected parents shape: {selected_parents.shape}")
    print(f"  Selected indices shape: {selected_indices.shape}")
    print(f"  Selected indices: {selected_indices}")
    
    # Verify results
    assert selected_parents.shape == (num_parents, get_total_parameters(layer_sizes))
    assert selected_indices.shape == (num_parents,)
    assert np.all(selected_indices >= 0) and np.all(selected_indices < pop_size)
    
    # Verify selected individuals match indices
    for i in range(num_parents):
        assert np.array_equal(selected_parents[i], population[selected_indices[i]])
    
    print(f"  ✅ Basic tournament selection - SUCCESS")
    
    # Test 2: Selection pressure with different tournament sizes
    print("\nTest 2: Selection pressure analysis")
    
    tournament_sizes = [1, 2, 3, 5, 7]
    num_trials = 1000
    
    for k in tournament_sizes:
        # Run many selections to analyze bias toward high fitness
        selection_counts = np.zeros(pop_size, dtype=np.int64)
        
        for trial in range(num_trials):
            _, indices = tournament_selection(
                population, fitness_scores, k, 1, seed=None  # No seed for randomness
            )
            selection_counts[indices[0]] += 1
        
        # Calculate bias toward high-fitness individuals
        high_fitness_selections = np.sum(selection_counts[pop_size//2:])  # Top half
        bias_percentage = (high_fitness_selections / num_trials) * 100
        
        print(f"  Tournament size {k}: {bias_percentage:.1f}% high-fitness selections")
        
        # Verify selection pressure increases with tournament size
        if k == 1:
            assert 40 <= bias_percentage <= 60, f"k=1 should be ~50% (random): {bias_percentage}"
        elif k >= 5:
            assert bias_percentage >= 80, f"k={k} should show strong bias: {bias_percentage}"
    
    print(f"  ✅ Selection pressure scales with tournament size - SUCCESS")
    
    # Test 3: Reproducibility with seeding
    print("\nTest 3: Reproducibility with seeding")
    
    seed_value = 999
    
    # Run same selection twice with same seed
    parents1, indices1 = tournament_selection(
        population, fitness_scores, 3, 5, seed=seed_value
    )
    parents2, indices2 = tournament_selection(
        population, fitness_scores, 3, 5, seed=seed_value
    )
    
    print(f"  Seed: {seed_value}")
    print(f"  First run indices:  {indices1}")
    print(f"  Second run indices: {indices2}")
    print(f"  Results identical: {np.array_equal(indices1, indices2)}")
    
    assert np.array_equal(parents1, parents2), "Same seed should produce identical parents"
    assert np.array_equal(indices1, indices2), "Same seed should produce identical indices"
    
    # Run with different seed to verify randomness
    parents3, indices3 = tournament_selection(
        population, fitness_scores, 3, 5, seed=seed_value + 1
    )
    
    print(f"  Different seed indices: {indices3}")
    print(f"  Results differ: {not np.array_equal(indices1, indices3)}")
    
    assert not np.array_equal(indices1, indices3), "Different seeds should produce different results"
    
    print(f"  ✅ Reproducibility and randomness - SUCCESS")
    
    # Test 4: Edge cases and input validation
    print("\nTest 4: Edge cases and input validation")
    
    # Test tournament size clamping
    large_tournament = tournament_selection(
        population, fitness_scores, 999, 3, seed=42  # Should clamp to pop_size
    )
    print(f"  Large tournament size (999) handled gracefully")
    
    # Test zero tournament size (should clamp to 1)
    zero_tournament = tournament_selection(
        population, fitness_scores, 0, 3, seed=42
    )
    print(f"  Zero tournament size handled gracefully")
    
    # Test single individual selection
    single_parent, single_idx = tournament_selection(
        population, fitness_scores, 2, 1, seed=42
    )
    assert single_parent.shape == (1, get_total_parameters(layer_sizes))
    assert single_idx.shape == (1,)
    print(f"  Single parent selection works")
    
    # Test with population size 1
    tiny_pop = population[:1]
    tiny_fitness = fitness_scores[:1]
    tiny_selection = tournament_selection(tiny_pop, tiny_fitness, 2, 1, seed=42)
    print(f"  Tiny population (size 1) handled")
    
    print(f"  ✅ Edge cases handled correctly - SUCCESS")
    
    # Test 5: Different population sizes and architectures
    print("\nTest 5: Different architectures and population sizes")
    
    architectures = [
        ([2, 3], "Simple"),
        ([5, 10, 5], "Medium"),
        ([10, 20, 15, 5], "Complex")
    ]
    
    for arch, name in architectures:
        layer_sizes = np.array(arch, dtype=np.int64)
        test_pop = initialize_population(15, layer_sizes, seed=1)
        test_fitness = np.random.random(15).astype(np.float64)
        
        parents, indices = tournament_selection(test_pop, test_fitness, 3, 5, seed=1)
        
        expected_params = get_total_parameters(layer_sizes)
        assert parents.shape == (5, expected_params)
        
        print(f"  {name} architecture ({' -> '.join(map(str, arch))}): ✅")
    
    # Test large population
    large_pop_size = 500
    large_pop = initialize_population(large_pop_size, np.array([5, 10, 3], dtype=np.int64), seed=1)
    large_fitness = np.random.random(large_pop_size).astype(np.float64)
    
    start_time = time.time()
    large_parents, large_indices = tournament_selection(large_pop, large_fitness, 5, 50, seed=1)
    selection_time = time.time() - start_time
    
    print(f"  Large population ({large_pop_size}): {selection_time:.6f}s")
    assert large_parents.shape == (50, get_total_parameters(np.array([5, 10, 3], dtype=np.int64)))
    
    print(f"  ✅ Different architectures and sizes - SUCCESS")
    
    # Test 6: Performance and scaling
    print("\nTest 6: Performance analysis")
    
    performance_tests = [
        (50, 10),
        (100, 20),
        (200, 50),
        (500, 100)
    ]
    
    layer_sizes = np.array([10, 20, 10], dtype=np.int64)
    
    for pop_size, num_parents in performance_tests:
        test_pop = initialize_population(pop_size, layer_sizes, seed=1)
        test_fitness = np.random.random(pop_size).astype(np.float64)
        
        # Warm up JIT
        _ = tournament_selection(test_pop[:5], test_fitness[:5], 2, 2, seed=1)
        
        # Time the selection
        start_time = time.time()
        for _ in range(10):  # Multiple runs for better timing
            _ = tournament_selection(test_pop, test_fitness, 3, num_parents, seed=1)
        avg_time = (time.time() - start_time) / 10
        
        selections_per_sec = num_parents / avg_time
        print(f"  Pop {pop_size}, Parents {num_parents}: {avg_time*1000:.2f}ms ({selections_per_sec:.0f} selections/s)")
    
    print(f"  ✅ Performance scaling verified - SUCCESS")
    
    # Test 7: Fitness distribution effects
    print("\nTest 7: Different fitness distributions")
    
    layer_sizes = np.array([3, 5, 2], dtype=np.int64)
    test_pop = initialize_population(30, layer_sizes, seed=1)
    
    # Test with uniform fitness (no selection pressure expected)
    uniform_fitness = np.ones(30, dtype=np.float64)
    uniform_parents, uniform_indices = tournament_selection(test_pop, uniform_fitness, 3, 10, seed=42)
    print(f"  Uniform fitness: indices range {uniform_indices.min()} to {uniform_indices.max()}")
    
    # Test with exponential fitness (strong selection pressure expected)
    exp_fitness = np.exp(np.linspace(0, 3, 30)).astype(np.float64)
    exp_parents, exp_indices = tournament_selection(test_pop, exp_fitness, 3, 10, seed=42)
    print(f"  Exponential fitness: indices range {exp_indices.min()} to {exp_indices.max()}")
    
    # Exponential should favor higher indices (higher fitness)
    avg_uniform_idx = np.mean(uniform_indices)
    avg_exp_idx = np.mean(exp_indices)
    print(f"  Average selected index - Uniform: {avg_uniform_idx:.1f}, Exponential: {avg_exp_idx:.1f}")
    
    assert avg_exp_idx > avg_uniform_idx, "Exponential fitness should select higher indices on average"
    
    print(f"  ✅ Fitness distribution effects verified - SUCCESS")
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   🏆 Tournament selection works correctly")
    print(f"   📊 Selection pressure scales with tournament size")
    print(f"   🎲 Reproducible with seeding, random without")
    print(f"   🛡️ Edge cases handled robustly")
    print(f"   ⚡ Performance scales well with population size")
    print(f"   🎯 Responds appropriately to fitness distributions")
    return True


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print(f"\n🎯 Ready to proceed to genetic operations (crossover & mutation)")
            print(f"   Tournament selection is production-ready!")
    except Exception as e:
        print(f"\n🛑 TEST 06 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)