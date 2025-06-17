# Time-Aware Neural Network Architecture Discussion

## Input Format
```python
inputs = (t, x)  # t = scalar timestamp, x = input vector (size = layer_sizes[0])
```

## Time-Aware Behavior
For the same input vector `x`:
- At time `t0 + 10` → output₁
- At time `t0 + 100` → output₂ (different due to time gap affecting decay)

## State Management Per Individual
```python
# Each individual maintains:
prev_states[individual_i] = [neuron₁_output, neuron₂_output, ...]  # Previous outputs
prev_time[individual_i] = last_evaluation_timestamp                # When last evaluated
```

## State Arrays
```python
population_states = np.zeros((pop_size, total_neurons), dtype=np.float64)
population_prev_times = np.zeros(pop_size, dtype=np.float64)
```

## Key Insight
Time decay affects memory: `exp(-(current_time - prev_time))`
Same input `x` at different times produces different outputs due to memory decay and previous internal states.

## Memory State Storage Decision: BEFORE Activation (Biological Accuracy)

### Chosen Approach (Option B):
```python
linear_out = np.dot(current_values, weights_matrix) + biases
memory_integrated = linear_out + alpha * prev_linear_states * decay
activated = apply_activation(memory_integrated)
# Store the pre-activation state (membrane potential analog)
prev_linear_states = memory_integrated  
```

### Formula:
```
linear = w^T * x + b
memory = linear + α * prev_linear * exp(-Δt)
output = f(memory)
store = memory  # Store pre-activation state
```

### Biological Justification:
- **Membrane potential** (pre-activation) persists over time in real neurons
- **Spike output** (post-activation) is brief and doesn't carry forward
- **Memory** = sustained membrane state, not spike output
- More biologically accurate than storing post-activation states