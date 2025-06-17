# Discrete-Time Leaky Integration for Time-Aware Neural Networks

## 🎯 **Core Concept**

We use **discrete-time leaky integration** to simulate continuous-time neural dynamics. This approach combines the simplicity of leaky integration with true time-awareness by discretizing time into small steps.

## 🧠 **The Method**

### **Standard Leaky Integration (Single Step):**
```python
new_state = alpha * prev_state + (1 - alpha) * new_input
```

### **Our Discrete-Time Extension:**
```python
# 1. Calculate time gap in discrete steps
time_diff = current_time - prev_time  # e.g., 0.015 seconds
resolution = 0.001  # 1ms resolution
num_steps = int(time_diff / resolution)  # e.g., 15 steps

# 2. Apply pure decay for each time step (no new input during gap)
decayed_state = prev_state * (alpha ** num_steps)

# 3. Then integrate new input
new_state = alpha * decayed_state + (1 - alpha) * new_input
```

## ⚡ **Mathematical Insight**

Between neural network evaluations, the neuron continues "running" with no input:
- **Step 1**: `state = alpha * prev_state` (pure decay)
- **Step 2**: `state = alpha * state = alpha² * prev_state`
- **Step n**: `state = alphaⁿ * prev_state`

So we can compute n decay steps instantly: `alpha ** num_steps`

## 📊 **Example Scenarios**

### **Rapid Inputs (1ms gap):**
```python
time_diff = 0.001  # 1ms
num_steps = 1
decay_factor = alpha ** 1 = alpha
# Behaves like standard leaky integration
```

### **Moderate Gap (10ms):**
```python
time_diff = 0.010  # 10ms  
num_steps = 10
decay_factor = alpha ** 10
# More decay than standard case
```

### **Long Gap (1 second):**
```python
time_diff = 1.0  # 1000ms
num_steps = 1000
decay_factor = alpha ** 1000
# Significant decay (approaches 0 if alpha < 1)
```

## 🔧 **Implementation Formula**

```python
def discrete_time_leaky_integration(prev_state, new_input, time_diff, alpha, resolution=0.001):
    """
    Apply discrete-time leaky integration with time-aware decay.
    
    Args:
        prev_state: Previous neuron state (pre-activation)
        new_input: Current linear input (w^T * x + b)
        time_diff: Time elapsed since last evaluation (seconds)
        alpha: Memory parameter [0,1] (0=no memory, 1=perfect memory)
        resolution: Time step resolution (default 1ms)
    
    Returns:
        new_state: Updated neuron state (pre-activation)
    """
    # Calculate discrete time steps
    num_steps = max(0, int(time_diff / resolution))
    
    # Apply temporal decay
    decayed_state = prev_state * (alpha ** num_steps)
    
    # Integrate new input
    new_state = alpha * decayed_state + (1 - alpha) * new_input
    
    return new_state
```

## 🎮 **Parameter Behavior**

### **Alpha (Memory Parameter):**
- `alpha = 1.0`: Perfect memory (no decay)
- `alpha = 0.9`: Slow decay (90% retention per step)
- `alpha = 0.5`: Medium decay (50% retention per step)  
- `alpha = 0.1`: Fast decay (10% retention per step)
- `alpha = 0.0`: No memory (immediate reset)

### **Resolution (Time Granularity):**
- `1ms` (0.001s): High precision, fine temporal control
- `10ms` (0.01s): Lower precision, coarser temporal effects
- Choice depends on timescales of interest in your application

## ✅ **Advantages**

1. **Time-Aware**: Different time gaps produce different outputs
2. **Simple Math**: Only multiplication and exponentiation (no `exp()`)
3. **Biologically Motivated**: Models actual neuron membrane dynamics
4. **Numerically Stable**: No overflow/underflow issues with reasonable alpha
5. **Intuitive Parameters**: Alpha directly controls memory strength
6. **Fast Computation**: `alpha ** n` is much faster than `exp(-t/tau)`

## 🔍 **Edge Cases**

### **Very Long Time Gaps:**
```python
# For alpha=0.9, time_diff=10 seconds (10,000 steps):
decay_factor = 0.9 ** 10000 ≈ 0  # Underflows to zero
# This is actually correct behavior - very old memories should fade
```

### **Zero Time Gap:**
```python
time_diff = 0.0
num_steps = 0  
decay_factor = alpha ** 0 = 1.0
# No temporal decay, just normal leaky integration
```

### **Sub-Resolution Time Gaps:**
```python
time_diff = 0.0005  # 0.5ms with 1ms resolution
num_steps = int(0.0005 / 0.001) = 0
# Treated as no time gap - acceptable for fine resolutions
```

## 🎯 **Usage in Neural Network**

```python
# For each neuron j in each layer:
linear_input = np.dot(layer_input, weights[:, j]) + biases[j]
new_state = discrete_time_leaky_integration(
    prev_states[j], 
    linear_input, 
    time_diff, 
    alphas[j]
)
output = apply_activation(new_state)
prev_states[j] = new_state  # Store pre-activation state
```

## 🔄 **Why This Works**

This approach simulates running a leaky integrator at high frequency (1kHz with 1ms resolution). Between neural network evaluations, the neuron "ticks" with no input, causing natural exponential-like decay while maintaining the simplicity of discrete leaky integration mathematics.

The result: **Time-aware neural networks with biologically motivated dynamics and computationally efficient implementation.**