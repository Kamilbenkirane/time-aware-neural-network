import numpy as np
import time
from typing import Callable, Tuple, List
import torch
import torch.multiprocessing as mp
from .individual import Individual


class GeneticAlgorithm:
    """Optimized genetic algorithm for neural network evolution."""
    
    def __init__(self, population_size=50, crossover_rate=0.9, mutation_rate=0.1, 
                 mutation_strength=0.1, tournament_size=3, elitism_count=2, 
                 individual_params=None, n_jobs=None):
        """Initialize genetic algorithm."""
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.individual_params = individual_params or {}
        
        # Parallel processing setup
        self.n_jobs = n_jobs or min(16, mp.cpu_count())  # Cap at 16 for your Mac
        self.pool = None  # Will create persistent pool for multiple evaluations
        
        # Set optimal thread count per process to avoid CPU oversubscription
        threads_per_process = max(1, mp.cpu_count() // self.n_jobs)
        torch.set_num_threads(threads_per_process)
        
        # Create population
        self.individuals = [Individual(**self.individual_params) for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)
        self.generation = 0
        self.rng = np.random.default_rng()
        
        print(f"🧬 Genetic Algorithm initialized (Population: {population_size})")
        if self.n_jobs > 1:
            print(f"🚀 Parallel processing: {self.n_jobs} workers, {threads_per_process} threads each")
    
    def _init_pool(self):
        """Initialize persistent process pool."""
        if self.pool is None and self.n_jobs > 1:
            self.pool = mp.Pool(self.n_jobs)
    
    def _close_pool(self):
        """Close process pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def evaluate_fitness(self, fitness_function):
        """Evaluate fitness for all individuals in parallel."""
        if self.n_jobs == 1:
            # Serial evaluation
            for i, individual in enumerate(self.individuals):
                self.fitness_scores[i] = fitness_function(individual)
        else:
            # Persistent pool parallel evaluation
            try:
                self._init_pool()
                scores = self.pool.map(fitness_function, self.individuals)
                self.fitness_scores = np.array(scores)
            except Exception as e:
                print(f"⚠️ Parallel failed, using serial: {str(e)[:50]}...")
                for i, individual in enumerate(self.individuals):
                    self.fitness_scores[i] = fitness_function(individual)
    
    def tournament_selection(self):
        """Tournament selection to choose parent."""
        indices = self.rng.choice(self.population_size, self.tournament_size, replace=False)
        winner_idx = indices[np.argmax(self.fitness_scores[indices])]
        return self.individuals[winner_idx]
    
    def crossover_mutate(self, parent1, parent2):
        """Combined crossover and mutation for efficiency."""
        w1, w2 = parent1.get_weights(), parent2.get_weights()
        
        # Crossover
        if self.rng.random() < self.crossover_rate:
            mask = self.rng.random(len(w1)) < 0.5
            child1_w = np.where(mask, w1, w2)
            child2_w = np.where(mask, w2, w1)
        else:
            child1_w, child2_w = w1.copy(), w2.copy()
        
        # Mutation
        for weights in [child1_w, child2_w]:
            mutation_mask = self.rng.random(len(weights)) < self.mutation_rate
            weights[mutation_mask] += self.rng.normal(0, self.mutation_strength, np.sum(mutation_mask))
        
        return (Individual(w0=child1_w, **self.individual_params),
                Individual(w0=child2_w, **self.individual_params))
    
    def evolve_generation(self):
        """Evolve one generation."""
        new_individuals = []
        
        # Elitism
        if self.elitism_count > 0:
            elite_indices = np.argsort(self.fitness_scores)[-self.elitism_count:]
            new_individuals.extend([self.individuals[i].clone() for i in elite_indices])
        
        # Generate offspring
        while len(new_individuals) < self.population_size:
            parent1, parent2 = self.tournament_selection(), self.tournament_selection()
            child1, child2 = self.crossover_mutate(parent1, parent2)
            new_individuals.extend([child1, child2])
        
        self.individuals = new_individuals[:self.population_size]
        self.generation += 1
    
    def get_stats(self):
        """Get current generation statistics."""
        return {
            'best_fitness': np.max(self.fitness_scores),
            'mean_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        }
    
    def run_evolution(self, fitness_function: Callable, generations=100, verbose=True):
        """Run genetic algorithm evolution."""
        stats_history = []
        start_time = time.time()
        
        if verbose:
            print(f"🚀 Starting evolution ({generations} generations)...")
        
        # Initial evaluation
        self.evaluate_fitness(fitness_function)
        
        for gen in range(generations):
            stats = self.get_stats()
            stats_history.append(stats)
            
            if verbose and gen % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Generation {gen:3d}: Best={stats['best_fitness']:8.4f}, "
                      f"Mean={stats['mean_fitness']:8.4f}, Std={stats['std_fitness']:.4f} "
                      f"[{elapsed:.1f}s]")
            
            self.evolve_generation()
            self.evaluate_fitness(fitness_function)
        
        # Final stats
        final_stats = self.get_stats()
        stats_history.append(final_stats)
        
        if verbose:
            total_time = time.time() - start_time
            print(f"🏁 Evolution complete! Total time: {total_time:.1f}s")
            print(f"   Final best fitness: {final_stats['best_fitness']:.4f}")
            print(f"   Avg time per generation: {total_time/generations:.2f}s")
        
        best_idx = np.argmax(self.fitness_scores)
        
        # Clean up pool
        self._close_pool()
        
        return self.individuals[best_idx], stats_history