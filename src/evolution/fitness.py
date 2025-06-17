import numpy as np
from typing import Callable
from .nb_individual import Individual

# Global variables for multiprocessing (fitness function state)
_global_prices = None
_global_timestamps = None
_global_normalized_prices = None
_global_risk_penalty = None
_global_sharpe_weight = None


def _parallel_trading_fitness(individual: Individual) -> float:
    """Parallel-safe trading fitness function using global variables."""
    individual.reset_state()
    portfolio_value = 1.0
    position = 0
    entry_price = 0.0
    trade_returns = []
    portfolio_values = [1.0]
    
    # Single pass - no tensor collection or replay
    for norm_price, price, timestamp in zip(_global_normalized_prices, _global_prices, _global_timestamps):
        action = individual.get_action_int(norm_price, timestamp)
        
        # Execute trading logic immediately
        if position == 0 and action == 2:  # Buy when no position
            position = 1
            entry_price = price
        elif position == 1 and action == 0:  # Sell when long
            position = 0
            trade_return = price / entry_price
            portfolio_value *= trade_return
            trade_returns.append(trade_return - 1.0)
        
        portfolio_values.append(portfolio_value)
    
    # Streamlined fitness calculation
    fitness = portfolio_value - 1.0  # Base return
    
    # Add Sharpe ratio if we have enough trades
    if len(trade_returns) > 1:
        returns_array = np.array(trade_returns)
        std_dev = np.std(returns_array)
        if std_dev > 0:
            fitness += _global_sharpe_weight * (np.mean(returns_array) / std_dev)
    
    # Drawdown penalty (only if severe)
    if len(portfolio_values) > 1:
        portfolio_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        max_drawdown = np.min((portfolio_array - running_max) / running_max)
        if max_drawdown < -0.2:  # >20% drawdown
            fitness += _global_risk_penalty * max_drawdown
    
    # Activity penalty
    if len(trade_returns) < 2:
        fitness -= 0.1
    
    return fitness


def create_trading_fitness_function(prices: np.ndarray, timestamps: np.ndarray, 
                                   risk_penalty: float = 0.1, sharpe_weight: float = 0.3) -> Callable:
    """Create optimized trading fitness function."""
    
    # Set global variables for multiprocessing
    global _global_prices, _global_timestamps, _global_normalized_prices
    global _global_risk_penalty, _global_sharpe_weight
    
    _global_prices = prices
    _global_timestamps = timestamps
    _global_normalized_prices = (prices - 1000) / 100
    _global_risk_penalty = risk_penalty
    _global_sharpe_weight = sharpe_weight
    
    def fitness_function(individual: Individual) -> float:
        """Single-pass trading simulation for maximum performance."""
        individual.reset_state()
        portfolio_value = 1.0
        position = 0  # 0=no position, 1=long
        entry_price = 0.0
        trade_returns = []
        portfolio_values = [1.0]
        
        # Single pass - no tensor collection or replay
        for norm_price, price, timestamp in zip(_global_normalized_prices, prices, timestamps):
            # Get action as integer directly (avoids tensor operations)
            action = individual.get_action_int(norm_price, timestamp)
            
            # Execute trading logic immediately
            if position == 0 and action == 2:  # Buy when no position
                position = 1
                entry_price = price
            elif position == 1 and action == 0:  # Sell when long
                position = 0
                trade_return = price / entry_price
                portfolio_value *= trade_return
                trade_returns.append(trade_return - 1.0)
            
            portfolio_values.append(portfolio_value)
        
        # Streamlined fitness calculation
        fitness = portfolio_value - 1.0  # Base return
        
        # Add Sharpe ratio if we have enough trades
        if len(trade_returns) > 1:
            returns_array = np.array(trade_returns)
            std_dev = np.std(returns_array)
            if std_dev > 0:
                fitness += sharpe_weight * (np.mean(returns_array) / std_dev)
        
        # Drawdown penalty (only if severe)
        if len(portfolio_values) > 1:
            portfolio_array = np.array(portfolio_values)
            running_max = np.maximum.accumulate(portfolio_array)
            max_drawdown = np.min((portfolio_array - running_max) / running_max)
            if max_drawdown < -0.2:  # >20% drawdown
                fitness += risk_penalty * max_drawdown
        
        # Activity penalty
        if len(trade_returns) < 2:
            fitness -= 0.1
        
        return fitness
    
    return fitness_function


def create_simple_test_fitness() -> Callable:
    """Simple fitness for algorithm validation."""
    def fitness_function(individual: Individual) -> float:
        weight_sum = np.sum(individual.get_weights())
        return 1.0 / (1.0 + abs(weight_sum - 10.0))
    return fitness_function


def create_sphere_function_fitness() -> Callable:
    """Sphere function for optimization benchmarking."""
    def fitness_function(individual: Individual) -> float:
        weights = individual.get_weights()
        return 1.0 / (1.0 + np.sum(weights ** 2))
    return fitness_function