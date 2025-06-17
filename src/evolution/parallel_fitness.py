"""Parallel-safe fitness functions for multiprocessing."""
import numpy as np
from .individual import Individual


class TradingFitnessEvaluator:
    """Parallel-safe trading fitness evaluator class."""
    
    def __init__(self, prices, timestamps, risk_penalty=0.1, sharpe_weight=0.3):
        self.prices = prices
        self.timestamps = timestamps
        self.normalized_prices = (prices - 1000) / 100
        self.risk_penalty = risk_penalty
        self.sharpe_weight = sharpe_weight
    
    def __call__(self, individual):
        """Evaluate trading fitness for an individual."""
        individual.reset_state()
        portfolio_value = 1.0
        position = 0
        entry_price = 0.0
        trade_returns = []
        portfolio_values = [1.0]
        
        # Single pass simulation
        for norm_price, price, timestamp in zip(self.normalized_prices, self.prices, self.timestamps):
            action = individual.get_action_int(norm_price, timestamp)
            
            if position == 0 and action == 2:  # Buy
                position = 1
                entry_price = price
            elif position == 1 and action == 0:  # Sell
                position = 0
                trade_return = price / entry_price
                portfolio_value *= trade_return
                trade_returns.append(trade_return - 1.0)
            
            portfolio_values.append(portfolio_value)
        
        # Calculate fitness
        fitness = portfolio_value - 1.0
        
        # Sharpe ratio
        if len(trade_returns) > 1:
            returns_array = np.array(trade_returns)
            std_dev = np.std(returns_array)
            if std_dev > 0:
                fitness += self.sharpe_weight * (np.mean(returns_array) / std_dev)
        
        # Drawdown penalty
        if len(portfolio_values) > 1:
            portfolio_array = np.array(portfolio_values)
            running_max = np.maximum.accumulate(portfolio_array)
            max_drawdown = np.min((portfolio_array - running_max) / running_max)
            if max_drawdown < -0.2:
                fitness += self.risk_penalty * max_drawdown
        
        # Activity penalty
        if len(trade_returns) < 2:
            fitness -= 0.1
        
        return fitness


def create_parallel_trading_fitness(prices, timestamps, risk_penalty=0.1, sharpe_weight=0.3):
    """Create a parallel-safe trading fitness function."""
    return TradingFitnessEvaluator(prices, timestamps, risk_penalty, sharpe_weight)