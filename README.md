# Time-Aware Neural Network Evolution

A high-performance genetic algorithm framework for evolving neural networks with temporal memory. Built with Numba JIT compilation for production-level performance and research applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [Performance](#performance)
- [Development Status](#development-status)
- [Contributing](#contributing)

## Features

- **Time-Aware Neural Networks**: Discrete-time leaky integration with learnable decay parameters
- **High-Performance Genetic Algorithms**: Tournament selection with 5M+ operations per second
- **Numba Optimized**: JIT compilation for production-level performance
- **Zero Dependencies**: Self-contained implementation (NumPy + Numba only)
- **Memory Efficient**: Pre-allocated arrays, cache-friendly access patterns
- **Research Ready**: Comprehensive test suite and educational notebooks

## Installation

### Requirements

- Python 3.8+
- NumPy
- Numba

### Install Dependencies

```bash
pip install numpy numba
```

### Clone Repository

```bash
git clone https://github.com/Kamilbenkirane/time-aware-neural-network.git
cd time-aware-neural-network
```

## Quick Start

### Basic Usage

```python
import numpy as np
from numba_ga import initialize_population, tournament_selection

# Create population of temporal neural networks
layer_sizes = np.array([4, 8, 3], dtype=np.int64)  # input -> hidden -> output
population = initialize_population(pop_size=100, layer_sizes=layer_sizes, seed=42)

# Example fitness evaluation (replace with your objective function)
fitness_scores = np.random.random(100).astype(np.float64)

# Tournament selection for genetic algorithm
selected_parents, parent_indices = tournament_selection(
    population, fitness_scores, tournament_size=2, num_parents=20, seed=123
)

print(f"Selected {len(selected_parents)} parents from population of {len(population)}")
```

### Time-Aware Prediction

```python
from numba_ga import predict_individual, reset_population_memory

# Initialize network network and memory
parameters = population[0]  # Use first individual
activations = np.array([1, 2], dtype=np.int64)  # ReLU -> Sigmoid
states, times = reset_population_memory(layer_sizes, 1)

# Sequential predictions with temporal memory
for t in range(10):
    inputs = (float(t), np.random.random(4))  # (timestamp, input_vector)
    outputs, states[0], times[0] = predict_individual(
        parameters, layer_sizes, activations, inputs, states[0], times[0], 
        *compute_layer_indices(layer_sizes)
    )
    print(f"Time {t}: Output = {outputs}")
```

## How It Works

### Time-Aware Neural Networks

Unlike standard neural networks that process inputs instantly, our networks maintain **temporal memory** using discrete-time leaky integration. Each neuron remembers its previous state and decays it over time.

#### Core Innovation: Discrete-Time Leaky Integration

Between neural network evaluations, neurons continue "running" with exponential decay:

```python
# Standard leaky integration (single step)
new_state = alpha * prev_state + (1 - alpha) * new_input

# Our time-aware extension
time_steps = int(time_gap / resolution_ms)  # e.g., 15ms gap = 15 steps  
decayed_state = prev_state * (alpha ** time_steps)  # Apply time-based decay
new_state = alpha * decayed_state + (1 - alpha) * new_input
```

#### Why This Matters

- **10ms gap**: `decay = alpha^10` (moderate memory fade)
- **1000ms gap**: `decay = alpha^1000` (significant memory fade)  
- **1ms gap**: `decay = alpha^1` (behaves like standard networks)

Each neuron learns its own `alpha` parameter (0=no memory, 1=perfect memory), creating networks that naturally handle temporal sequences.

## Architecture

### Neural Network Structure

- **Parameters per neuron**: Weights + bias + temporal alpha
- **State management**: Pre-activation states maintained across time
- **Activation functions**: ReLU, Sigmoid, Tanh, Leaky ReLU, Linear

### Genetic Algorithm Components

- **Population Management**: Batch initialization and memory state handling
- **Tournament Selection**: Configurable selection pressure (k=1 to k=7)
- **Parameter Structure**: Weights + biases + temporal alphas for each connection

### Performance Optimizations

- **Numba JIT**: All critical functions compiled with `@njit(fastmath=True, cache=True)`
- **Memory Pre-allocation**: Zero allocation in evolution loops
- **Cache-Friendly Access**: Pre-computed indices and reused buffers

## Examples

See the `Notebooks/demo/` directory for comprehensive examples:

- `06_population_fitness.ipynb`: Population-level fitness evaluation
- `07_tournament_selection.ipynb`: Selection pressure analysis and performance

## Performance

| Operation | Throughput | Configuration |
|-----------|------------|---------------|
| Tournament Selection | 5M+ selections/sec | Binary tournament (k=2) |
| Population Prediction | 970+ individuals/sec | 100 individuals, [4,8,3] architecture |
| Parameter Initialization | 10M+ parameters/sec | Normal distribution sampling |

*Benchmarks run on standard hardware with Numba JIT warmup*

## Development Status

### ✅ Completed
- Time-aware neural network prediction
- Population initialization and memory management
- Tournament selection with configurable pressure
- Comprehensive test suite (7 test categories)

### 🔄 In Progress
- Crossover operations (single-point, uniform, arithmetic)
- Mutation strategies (Gaussian, adaptive rates)
- Complete evolutionary loop

### 📋 Planned
- Alternative selection methods (roulette wheel, rank-based)
- Multi-epoch training capabilities
- Advanced genetic operators

## Contributing

We follow a research → implement → test pattern:

1. **Research**: Investigate best practices and algorithms
2. **Implement**: Write production-quality Numba-optimized code
3. **Test**: Create comprehensive test suites with performance benchmarks

See `todos.md` for current development priorities.

## License

This project is open source. See LICENSE file for details.