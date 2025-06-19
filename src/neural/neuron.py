import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeAwareLinear(nn.Module):
    """Optimized time-aware linear layer for genetic algorithms."""
    
    def __init__(self, in_features, out_features, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
        # Learnable network
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.3)
        self.bias = nn.Parameter(torch.randn(out_features) * 0.3)
        
        # Temporal state
        self.register_buffer('prev_values', torch.zeros(out_features))
        self.prev_timestamp = 0.0
        self.initialized = False
    
    def forward(self, x, current_time):
        # Ensure 2D input [1, in_features]
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Compute decay (use math.exp to avoid tensor creation)
        if self.initialized:
            time_diff = max(0.0, min(current_time - self.prev_timestamp, 50.0))
            decay = math.exp(-time_diff)  # Use math.exp instead of torch.exp
        else:
            decay = 0.0
            self.initialized = True
        
        # Linear + temporal memory
        linear_out = F.linear(x, self.weight, self.bias)
        memory_out = linear_out + self.alpha * self.prev_values * decay
        output = torch.tanh(memory_out)
        
        # Update state
        self.prev_values.copy_(output[0])
        self.prev_timestamp = current_time
        
        return output[0] if x.size(0) == 1 else output
    
    def reset_state(self):
        self.prev_values.zero_()
        self.prev_timestamp = 0.0
        self.initialized = False


class TimeAwareNetwork(nn.Module):
    """Optimized time-aware network."""
    
    def __init__(self, input_size=1, hidden_size=10, output_size=3, alpha=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = TimeAwareLinear(input_size, hidden_size, alpha)
        self.output_layer = TimeAwareLinear(hidden_size, output_size, alpha)
    
    def forward(self, x, current_time):
        hidden = self.hidden_layer(x, current_time)
        return self.output_layer(hidden, current_time)
    
    def get_action(self, x, current_time):
        with torch.no_grad():
            return torch.argmax(self.forward(x, current_time))
    
    def reset_state(self):
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()
    
    def get_weights_flat(self):
        return torch.cat([
            self.hidden_layer.weight.data.flatten(),
            self.hidden_layer.bias.data,
            self.output_layer.weight.data.flatten(),
            self.output_layer.bias.data
        ])
    
    def set_weights_flat(self, weights):
        idx = 0
        # Hidden layer weights + bias
        hidden_w_size = self.hidden_size * self.input_size
        self.hidden_layer.weight.data = weights[idx:idx + hidden_w_size].view(self.hidden_size, self.input_size)
        idx += hidden_w_size
        self.hidden_layer.bias.data = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        # Output layer weights + bias
        output_w_size = self.output_size * self.hidden_size
        self.output_layer.weight.data = weights[idx:idx + output_w_size].view(self.output_size, self.hidden_size)
        idx += output_w_size
        self.output_layer.bias.data = weights[idx:idx + self.output_size]
    
    @property
    def total_weights(self):
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.output_size + self.output_size)