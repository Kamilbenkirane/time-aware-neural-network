"""
Optimal trading fitness functions for genetic algorithms.
Designed for maximum performance using Numba best practices.
"""

import numpy as np
from numba import njit, prange
import sys
import os

# Add parent directory to path for numba_ga import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numba_ga


@njit(cache=True, fastmath=True)
def evaluate_individual_fitness_relative(parameters, layer_sizes, activations,
                                        timestamps, prices_normalized,
                                        param_indices, neuron_indices):
    """
    Phase 1: Evaluate trading fitness with relative normalization by first price.
    
    Scale-invariant approach that preserves all relative relationships.
    Works for any price range without hardcoded constants.
    """
    # Pre-allocate reused arrays (thread-local)
    input_buffer = np.zeros(1, dtype=np.float64)
    current_states = np.zeros(np.sum(layer_sizes[1:]), dtype=np.float64)
    
    # Trading state (all in normalized space)
    position = 0  # 0=cash, 1=stock
    buy_price_norm = 0.0
    portfolio_value = 1.0
    current_time = 0.0
    
    # Main evaluation loop
    for i in range(len(timestamps)):
        # Reuse input buffer (zero allocation)
        input_buffer[0] = prices_normalized[i]
        inputs = (timestamps[i], input_buffer)
        
        # Network prediction with temporal memory
        output, new_states, new_time = numba_ga.predict_individual(
            parameters, layer_sizes, activations, inputs,
            current_states, current_time, param_indices, neuron_indices
        )
        
        # Optimized argmax for 3 elements (faster than np.argmax)
        action = (0 if output[0] >= output[1] and output[0] >= output[2] 
                 else 1 if output[1] >= output[2] else 2)
        
        # Trading logic in normalized space (ratios preserved)
        if position == 0 and action == 2:  # Cash → Buy
            position = 1
            buy_price_norm = prices_normalized[i]
        elif position == 1 and action == 0:  # Stock → Sell
            sell_price_norm = prices_normalized[i]
            portfolio_value *= (sell_price_norm / buy_price_norm)
            position = 0
        
        # Update network state
        current_states = new_states
        current_time = new_time
    
    # Close position if holding stock at epoch end
    if position == 1:
        final_price_norm = prices_normalized[-1]
        portfolio_value *= (final_price_norm / buy_price_norm)
    
    return portfolio_value


@njit(cache=True, fastmath=True)
def evaluate_individual_fitness_returns(parameters, layer_sizes, activations,
                                       timestamps, price_returns,
                                       param_indices, neuron_indices):
    """
    Phase 2: Evaluate trading fitness using price returns.
    
    Feeds percentage changes to network - more predictive signal,
    stationary time series, naturally bounded inputs.
    """
    # Pre-allocate reused arrays (thread-local)
    input_buffer = np.zeros(1, dtype=np.float64)
    current_states = np.zeros(np.sum(layer_sizes[1:]), dtype=np.float64)
    
    # Trading state (tracking relative performance)
    position = 0  # 0=cash, 1=stock
    cumulative_return = 1.0  # Track compound returns
    portfolio_value = 1.0
    current_time = 0.0
    
    # Main evaluation loop
    for i in range(len(timestamps)):
        # Reuse input buffer (zero allocation)
        input_buffer[0] = price_returns[i]
        inputs = (timestamps[i], input_buffer)
        
        # Network prediction with temporal memory
        output, new_states, new_time = numba_ga.predict_individual(
            parameters, layer_sizes, activations, inputs,
            current_states, current_time, param_indices, neuron_indices
        )
        
        # Optimized argmax for 3 elements (faster than np.argmax)
        action = (0 if output[0] >= output[1] and output[0] >= output[2] 
                 else 1 if output[1] >= output[2] else 2)
        
        # Update cumulative return for current tick
        cumulative_return *= (1.0 + price_returns[i])
        
        # Trading logic based on position exposure
        if position == 0 and action == 2:  # Cash → Buy (enter position)
            position = 1
            portfolio_value = cumulative_return  # Take current market level
        elif position == 1 and action == 0:  # Stock → Sell (exit position)
            position = 0
            # Portfolio grows with market while holding, stays flat when cash
        # When holding stock: portfolio tracks cumulative_return
        # When holding cash: portfolio stays at last exit level
        
        if position == 1:
            portfolio_value = cumulative_return
    
    return portfolio_value


@njit(parallel=True, cache=True, fastmath=True)
def evaluate_population_fitness_relative(population, layer_sizes, activations,
                                        timestamps, prices):
    """
    Phase 1: Evaluate population with relative normalization by first price.
    
    Scale-invariant parallel evaluation for any price range.
    """
    pop_size = population.shape[0]
    
    # Pre-allocate result array
    fitness_results = np.zeros(pop_size, dtype=np.float64)
    
    # Pre-compute shared data once (avoid recomputation in parallel loops)
    param_indices, neuron_indices = numba_ga.compute_layer_indices(layer_sizes)
    prices_normalized = prices / prices[0]  # Relative to first price
    
    # Parallel evaluation across population
    for i in prange(pop_size):
        fitness_results[i] = evaluate_individual_fitness_relative(
            population[i], layer_sizes, activations,
            timestamps, prices_normalized,
            param_indices, neuron_indices
        )
    
    return fitness_results


@njit(parallel=True, cache=True, fastmath=True)
def evaluate_population_fitness_returns(population, layer_sizes, activations,
                                       timestamps, prices):
    """
    Phase 2: Evaluate population using price returns.
    
    Feeds percentage changes - more predictive and stationary.
    """
    pop_size = population.shape[0]
    
    # Pre-allocate result array
    fitness_results = np.zeros(pop_size, dtype=np.float64)
    
    # Pre-compute shared data once
    param_indices, neuron_indices = numba_ga.compute_layer_indices(layer_sizes)
    
    # Compute price returns (vectorized)
    price_returns = np.zeros(len(prices), dtype=np.float64)
    price_returns[0] = 0.0  # No previous price for first tick
    for j in range(1, len(prices)):
        price_returns[j] = (prices[j] - prices[j-1]) / prices[j-1]
    
    # Parallel evaluation across population
    for i in prange(pop_size):
        fitness_results[i] = evaluate_individual_fitness_returns(
            population[i], layer_sizes, activations,
            timestamps, price_returns,
            param_indices, neuron_indices
        )
    
    return fitness_results


# Convenience aliases for backward compatibility and easy switching
evaluate_individual_fitness = evaluate_individual_fitness_relative
evaluate_population_fitness = evaluate_population_fitness_relative