import torch
from src.neural.neuron import TimeAwareNetwork


class Individual:
    """Evolutionary individual wrapping a TimeAwareNetwork."""
    
    def __init__(self, w0=None, input_size=1, hidden_size=10, output_size=3, alpha=1.0):
        """
        Initialize individual with neural network weights.
        
        Args:
            w0: 1D array/tensor of weights, if None then random
            input_size: Number of input features (default: 1)
            hidden_size: Number of hidden neurons (default: 10)
            output_size: Number of output actions (default: 3)
            alpha: Temporal decay parameter (default: 1.0)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.network = TimeAwareNetwork(input_size, hidden_size, output_size, alpha)
        self._input_tensor = torch.zeros(1, dtype=torch.float32)
        
        if w0 is not None:
            self.set_weights(w0)
        self.w0 = self.get_weights()  # Cache initial weights
    
    def get_action(self, price_info, current_time):
        """Get action from neural network."""
        self._input_tensor.fill_(price_info)
        return self.network.get_action(self._input_tensor, current_time)
    
    def get_action_int(self, price_info, current_time):
        """Get action as integer directly for maximum performance."""
        self._input_tensor.fill_(price_info)
        with torch.no_grad():
            output = self.network.forward(self._input_tensor, current_time)
            return int(torch.argmax(output).item())
    
    def reset_state(self):
        """Reset temporal memory."""
        self.network.reset_state()
    
    @property
    def total_weights(self):
        """Total number of weights in the network."""
        return self.network.total_weights
    
    def get_weights(self):
        """Get current weights as numpy array."""
        return self.network.get_weights_flat().detach().numpy()
    
    def set_weights(self, weights):
        """Set network weights."""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.network.set_weights_flat(weights)
        self.w0 = weights.detach().numpy() if isinstance(weights, torch.Tensor) else weights
    
    def clone(self):
        """Create a deep copy of this individual."""
        return Individual(
            w0=self.w0.copy(),
            input_size=self.input_size,
            hidden_size=self.hidden_size, 
            output_size=self.output_size,
            alpha=self.alpha
        )
    
    def __repr__(self):
        """String representation."""
        return f"Individual(weights={self.total_weights}, " \
               f"arch={self.input_size}-{self.hidden_size}-{self.output_size})"