"""NEAT: Neural Evolution for Automated Trading"""

from src.neural.neuron import TimeAwareLinear, TimeAwareNetwork
from src.evolution.individual import Individual

__all__ = ['TimeAwareLinear', 'TimeAwareNetwork', 'Individual']