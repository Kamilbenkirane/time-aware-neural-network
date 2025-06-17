"""
Pure Function-Based Fitness Functions with Numba JIT Compilation.

This module implements ultra-fast fitness functions using ONLY pure @njit functions
with raw numpy arrays. No classes, no object instantiation, no Python overhead.

Architecture:
Raw Weight Array → @njit Functions → Fitness Score

Key principles:
- Zero classes in core computation
- Raw array operations only
- Stateless pure functions
- Minimal memory allocation
- Maximum @njit optimization
"""

import numpy as np
import math
from numba import njit, prange
from typing import Callable


# ============================================================================
# WEIGHT ARRAY INDEXING FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def calculate_weight_indices(input_size, hidden_size, output_size):
    """
    Calculate array slice indices for weight array structure.
    
    Weight array layout:
    [hidden_weights, hidden_bias, output_weights, output_bias]
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Tuple of (hw_start, hw_end, hb_start, hb_end, ow_start, ow_end, ob_start, ob_end)
    """
    # Hidden weights: hidden_size × input_size
    hw_start = 0
    hw_end = hidden_size * input_size
    
    # Hidden bias: hidden_size
    hb_start = hw_end
    hb_end = hb_start + hidden_size
    
    # Output weights: output_size × hidden_size
    ow_start = hb_end
    ow_end = ow_start + output_size * hidden_size
    
    # Output bias: output_size
    ob_start = ow_end
    ob_end = ob_start + output_size
    
    return hw_start, hw_end, hb_start, hb_end, ow_start, ow_end, ob_start, ob_end


@njit(fastmath=True, cache=True)
def extract_weights(weight_array, input_size, hidden_size, output_size):
    """
    Extract weight matrices from flat weight array.
    
    Args:
        weight_array: Flat 1D weight array
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Tuple of (hidden_weights, hidden_bias, output_weights, output_bias)
    """
    # Calculate expected total size
    expected_size = input_size * hidden_size + hidden_size + output_size * hidden_size + output_size
    
    # Verify array size matches expectations
    if len(weight_array) != expected_size:
        # For debugging - create arrays of correct size with available weights
        available_weights = len(weight_array)
        # Fallback: create minimal valid arrays
        hidden_weights = np.zeros((hidden_size, input_size), dtype=np.float64)
        hidden_bias = np.zeros(hidden_size, dtype=np.float64)
        output_weights = np.zeros((output_size, hidden_size), dtype=np.float64)
        output_bias = np.zeros(output_size, dtype=np.float64)
        
        # Fill with available weights as much as possible
        idx = 0
        hw_size = hidden_size * input_size
        if idx + hw_size <= available_weights:
            hidden_weights = weight_array[idx:idx + hw_size].reshape(hidden_size, input_size)
            idx += hw_size
        
        if idx + hidden_size <= available_weights:
            hidden_bias = weight_array[idx:idx + hidden_size]
            idx += hidden_size
        
        ow_size = output_size * hidden_size
        if idx + ow_size <= available_weights:
            output_weights = weight_array[idx:idx + ow_size].reshape(output_size, hidden_size)
            idx += ow_size
        
        if idx + output_size <= available_weights:
            output_bias = weight_array[idx:idx + output_size]
        
        return hidden_weights, hidden_bias, output_weights, output_bias
    
    hw_start, hw_end, hb_start, hb_end, ow_start, ow_end, ob_start, ob_end = calculate_weight_indices(
        input_size, hidden_size, output_size
    )
    
    # Extract and reshape weights
    hidden_weights = weight_array[hw_start:hw_end].reshape(hidden_size, input_size)
    hidden_bias = weight_array[hb_start:hb_end]
    output_weights = weight_array[ow_start:ow_end].reshape(output_size, hidden_size)
    output_bias = weight_array[ob_start:ob_end]
    
    return hidden_weights, hidden_bias, output_weights, output_bias


# ============================================================================
# NEURAL NETWORK CORE FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def create_nn_state(hidden_size, output_size):
    """
    Create initial neural network temporal state.
    
    Args:
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
    
    Returns:
        Tuple of (hidden_prev, hidden_time, output_prev, output_time)
    """
    hidden_prev = np.zeros(hidden_size, dtype=np.float64)
    hidden_time = 0.0
    output_prev = np.zeros(output_size, dtype=np.float64)
    output_time = 0.0
    
    return hidden_prev, hidden_time, output_prev, output_time


@njit(fastmath=True, cache=True)
def reset_nn_state(hidden_prev, output_prev):
    """
    Reset neural network temporal state.
    
    Args:
        hidden_prev: Hidden layer previous values (modified in-place)
        output_prev: Output layer previous values (modified in-place)
    
    Returns:
        Reset timestamps (hidden_time=0.0, output_time=0.0)
    """
    hidden_prev.fill(0.0)
    output_prev.fill(0.0)
    return 0.0, 0.0


@njit(fastmath=True, cache=True)
def compute_temporal_decay(current_time, prev_time):
    """
    Compute exponential temporal decay factor.
    
    Args:
        current_time: Current timestamp
        prev_time: Previous timestamp
    
    Returns:
        Decay factor between 0 and 1
    """
    if prev_time <= 0.0:
        return 0.0
    
    time_diff = max(0.0, min(current_time - prev_time, 50.0))
    return math.exp(-time_diff)


@njit(fastmath=True, cache=True)
def layer_forward_pass(x, weights, bias, prev_values, prev_time, current_time, alpha):
    """
    Forward pass through a single time-aware layer.
    
    Args:
        x: Input array
        weights: Weight matrix (output_size, input_size)
        bias: Bias array (output_size,)
        prev_values: Previous layer outputs (modified in-place)
        prev_time: Previous timestamp
        current_time: Current timestamp
        alpha: Memory strength parameter
    
    Returns:
        Tuple of (output, new_prev_time)
    """
    # Linear transformation
    linear_out = np.dot(weights, x) + bias
    
    # Temporal decay
    decay_factor = compute_temporal_decay(current_time, prev_time)
    
    # Add temporal memory
    memory_contribution = alpha * prev_values * decay_factor
    memory_out = linear_out + memory_contribution
    
    # Apply activation
    output = np.tanh(memory_out)
    
    # Update previous values in-place
    prev_values[:] = output
    
    return output, current_time


@njit(fastmath=True, cache=True)
def nn_forward_pass(weight_array, input_value, current_time, nn_state, 
                   input_size, hidden_size, output_size, alpha):
    """
    Complete neural network forward pass with raw arrays.
    
    Args:
        weight_array: Flat weight array
        input_value: Single input value
        current_time: Current timestamp
        nn_state: Neural network state (hidden_prev, hidden_time, output_prev, output_time)
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons
        alpha: Memory strength parameter
    
    Returns:
        Tuple of (action_index, updated_nn_state)
    """
    hidden_prev, hidden_time, output_prev, output_time = nn_state
    
    # Extract weights
    hidden_weights, hidden_bias, output_weights, output_bias = extract_weights(
        weight_array, input_size, hidden_size, output_size
    )
    
    # Prepare input
    x_input = np.array([input_value], dtype=np.float64)
    
    # Hidden layer forward pass
    hidden_out, new_hidden_time = layer_forward_pass(
        x_input, hidden_weights, hidden_bias, hidden_prev, hidden_time, current_time, alpha
    )
    
    # Output layer forward pass
    output_out, new_output_time = layer_forward_pass(
        hidden_out, output_weights, output_bias, output_prev, output_time, current_time, alpha
    )
    
    # Get action (argmax)
    action_index = np.argmax(output_out)
    
    # Return updated state
    updated_state = (hidden_prev, new_hidden_time, output_prev, new_output_time)
    
    return action_index, updated_state


# ============================================================================
# TRADING SIMULATION FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def trading_simulation_pure(weight_array, prices, timestamps, normalized_prices,
                           input_size, hidden_size, output_size, alpha):
    """
    Pure function-based trading simulation using only raw arrays.
    
    Args:
        weight_array: Flat neural network weights
        prices: Price data array
        timestamps: Timestamp array
        normalized_prices: Pre-normalized price array
        input_size: Neural network input size
        hidden_size: Neural network hidden size
        output_size: Neural network output size
        alpha: Temporal memory strength
    
    Returns:
        Tuple of (portfolio_value, trade_returns, portfolio_values)
    """
    # Initialize neural network state
    nn_state = create_nn_state(hidden_size, output_size)
    
    # Trading state
    portfolio_value = 1.0
    position = 0  # 0=no position, 1=long
    entry_price = 0.0
    
    # Pre-allocate arrays
    max_trades = len(prices) // 5  # Conservative estimate
    trade_returns = np.empty(max_trades, dtype=np.float64)
    portfolio_values = np.empty(len(prices) + 1, dtype=np.float64)
    
    trade_count = 0
    portfolio_values[0] = 1.0
    
    # Main trading loop
    for i in range(len(prices)):
        price = prices[i]
        timestamp = timestamps[i]
        norm_price = normalized_prices[i]
        
        # Get action from neural network
        action, nn_state = nn_forward_pass(
            weight_array, norm_price, timestamp, nn_state,
            input_size, hidden_size, output_size, alpha
        )
        
        # Execute trading logic
        if position == 0 and action == 2:  # Buy when no position
            position = 1
            entry_price = price
        elif position == 1 and action == 0:  # Sell when long
            position = 0
            trade_return = price / entry_price
            portfolio_value *= trade_return
            
            if trade_count < max_trades:
                trade_returns[trade_count] = trade_return - 1.0
                trade_count += 1
        
        portfolio_values[i + 1] = portfolio_value
    
    # Return actual arrays (not pre-allocated size)
    actual_trade_returns = trade_returns[:trade_count]
    
    return portfolio_value, actual_trade_returns, portfolio_values


@njit(fastmath=True, cache=True)
def calculate_fitness_components(portfolio_value, trade_returns, portfolio_values,
                                risk_penalty, sharpe_weight):
    """
    Calculate fitness components from trading results.
    
    Args:
        portfolio_value: Final portfolio value
        trade_returns: Array of trade returns
        portfolio_values: Array of portfolio values over time
        risk_penalty: Risk penalty factor
        sharpe_weight: Sharpe ratio weight
    
    Returns:
        Total fitness score
    """
    # Base fitness (total return)
    base_fitness = portfolio_value - 1.0
    
    # Sharpe ratio bonus
    sharpe_bonus = 0.0
    if len(trade_returns) > 1:
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        if std_return > 0:
            sharpe_bonus = sharpe_weight * (mean_return / std_return)
    
    # Drawdown penalty
    drawdown_penalty = 0.0
    if len(portfolio_values) > 1:
        running_max = portfolio_values[0]
        max_drawdown = 0.0
        
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i] > running_max:
                running_max = portfolio_values[i]
            
            drawdown = (portfolio_values[i] - running_max) / running_max
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        if max_drawdown < -0.2:  # >20% drawdown
            drawdown_penalty = risk_penalty * max_drawdown
    
    # Activity penalty
    activity_penalty = 0.0
    if len(trade_returns) < 2:
        activity_penalty = -0.1
    
    return base_fitness + sharpe_bonus + drawdown_penalty + activity_penalty


@njit(fastmath=True, cache=True)
def trading_fitness_core(weight_array, prices, timestamps, normalized_prices,
                        risk_penalty, sharpe_weight, input_size, hidden_size, output_size, alpha):
    """
    Core trading fitness function using pure @njit functions only.
    
    Args:
        weight_array: Flat neural network weights
        prices: Price data
        timestamps: Timestamp data
        normalized_prices: Pre-normalized prices
        risk_penalty: Risk penalty factor
        sharpe_weight: Sharpe ratio weight
        input_size: Neural network input size
        hidden_size: Neural network hidden size
        output_size: Neural network output size
        alpha: Temporal memory strength
    
    Returns:
        Fitness score (float)
    """
    # Run trading simulation
    portfolio_value, trade_returns, portfolio_values = trading_simulation_pure(
        weight_array, prices, timestamps, normalized_prices,
        input_size, hidden_size, output_size, alpha
    )
    
    # Calculate fitness
    return calculate_fitness_components(
        portfolio_value, trade_returns, portfolio_values,
        risk_penalty, sharpe_weight
    )


# ============================================================================
# BENCHMARK FITNESS FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def sphere_fitness_core(weight_array):
    """
    Pure sphere function using raw weight array.
    
    Args:
        weight_array: Flat weight array
    
    Returns:
        Negative sum of squares (for maximization)
    """
    return -np.sum(weight_array * weight_array)


@njit(fastmath=True, cache=True)
def simple_test_fitness_core(weight_array, target_sum):
    """
    Pure simple test fitness using raw weight array.
    
    Args:
        weight_array: Flat weight array
        target_sum: Target weight sum
    
    Returns:
        Fitness score
    """
    weight_sum = np.sum(weight_array)
    return 1.0 / (1.0 + abs(weight_sum - target_sum))


# ============================================================================
# API WRAPPER FUNCTIONS (MINIMAL - NO CLASSES)
# ============================================================================

def create_pure_trading_fitness_function(prices: np.ndarray, timestamps: np.ndarray,
                                        risk_penalty: float = 0.1, sharpe_weight: float = 0.3,
                                        input_size: int = 1, hidden_size: int = 10, 
                                        output_size: int = 3, alpha: float = 1.0) -> Callable:
    """
    Create pure function-based trading fitness function.
    
    Args:
        prices: Price data array
        timestamps: Timestamp array
        risk_penalty: Risk penalty factor
        sharpe_weight: Sharpe ratio weight
        input_size: Neural network input size
        hidden_size: Neural network hidden size
        output_size: Neural network output size
        alpha: Temporal memory strength
    
    Returns:
        Fitness function that takes (individual) and returns fitness
    """
    # Pre-compute normalized prices
    normalized_prices = (prices - 1000.0) / 100.0
    
    # Ensure arrays are contiguous and correct dtype
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    timestamps = np.ascontiguousarray(timestamps, dtype=np.float64)
    normalized_prices = np.ascontiguousarray(normalized_prices, dtype=np.float64)
    
    def fitness_function(individual) -> float:
        """Pure fitness function - only extracts weights, rest is pure @njit."""
        weight_array = individual.get_weights()
        
        return trading_fitness_core(
            weight_array, prices, timestamps, normalized_prices,
            risk_penalty, sharpe_weight, input_size, hidden_size, output_size, alpha
        )
    
    return fitness_function


def create_pure_sphere_fitness_function() -> Callable:
    """Create pure sphere fitness function."""
    def fitness_function(individual) -> float:
        weight_array = individual.get_weights()
        return sphere_fitness_core(weight_array)
    
    return fitness_function


def create_pure_simple_test_fitness_function(target_sum: float = 10.0) -> Callable:
    """Create pure simple test fitness function."""
    def fitness_function(individual) -> float:
        weight_array = individual.get_weights()
        return simple_test_fitness_core(weight_array, target_sum)
    
    return fitness_function


# ============================================================================
# COMPATIBILITY ALIASES
# ============================================================================

# Drop-in replacements for existing functions
create_trading_fitness_function = create_pure_trading_fitness_function
create_sphere_function_fitness = create_pure_sphere_fitness_function
create_simple_test_fitness = create_pure_simple_test_fitness_function


# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

def benchmark_pure_fitness_performance(population_size=100, data_points=2000):
    """
    Benchmark pure function fitness performance.
    
    Args:
        population_size: Number of individuals to test
        data_points: Number of price data points
    
    Returns:
        Performance metrics
    """
    import time

    
    # Generate test data
    np.random.seed(42)
    prices = 1000.0 + np.random.randn(data_points) * 50
    timestamps = np.arange(data_points, dtype=np.float64)
    
    # Create fitness function
    fitness_func = create_pure_trading_fitness_function(prices, timestamps)
    
    # Create test population
    individuals = [NumbaIndividual() for _ in range(population_size)]
    
    # Warm up compilation
    for i in range(3):
        _ = fitness_func(individuals[0])
    
    # Benchmark evaluation
    start_time = time.time()
    
    for individual in individuals:
        _ = fitness_func(individual)
    
    evaluation_time = time.time() - start_time
    
    print(f"Pure Function Fitness Performance:")
    print(f"Population size: {population_size}")
    print(f"Data points: {data_points}")
    print(f"Total evaluation time: {evaluation_time:.4f} seconds")
    print(f"Time per evaluation: {evaluation_time/population_size*1000:.4f} ms")
    print(f"Evaluations per second: {population_size/evaluation_time:.0f}")
    
    return {
        'total_time': evaluation_time,
        'time_per_evaluation': evaluation_time / population_size,
        'evaluations_per_second': population_size / evaluation_time
    }


