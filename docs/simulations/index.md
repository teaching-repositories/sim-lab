# SimLab Simulations

SimLab offers a comprehensive collection of simulation tools for modeling complex systems across different domains. All simulators follow a consistent interface pattern, are statically typed, and provide robust validation and error handling.

## Simulation Categories

Our simulations are organized into the following categories:

### Basic Simulations
- [Stock Market Simulation](basic/stock_market.md): Model stock price fluctuations with factors like volatility, drift, and market events
- [Resource Fluctuations Simulation](basic/resource_fluctuations.md): Simulate resource price dynamics with supply disruptions
- [Product Popularity Simulation](basic/product_popularity.md): Model product demand considering growth, marketing, and promotions
- [Modelling Market Dynamics](basic/modelling_market_dynamics.md): Theory behind our market dynamics models

### Discrete Event Simulations
- [Discrete Event Simulation](discrete_event/discrete_event.md): General-purpose event-driven simulation engine
- **Queueing Simulation**: Model service systems with arrivals, queues, and servers (coming soon)

### Statistical Simulations
- **Monte Carlo Simulation**: Sample random processes to estimate numerical results (coming soon)
- **Markov Chain Simulation**: Model stochastic processes with the Markov property (coming soon)

### Agent-Based Simulations
- [Agent-Based Simulation](agent_based/agent_based.md): Model complex systems through interactions of autonomous agents

### System Dynamics
- **System Dynamics Simulation**: Model systems with stocks, flows, and feedback loops (coming soon)

### Network Simulations
- [Network Simulation](network/network.md): Model processes on complex networks with different topologies

### Ecological Simulations
- **Predator-Prey Simulation**: Model population dynamics using Lotka-Volterra equations (coming soon)

### Domain-Specific Simulations
- **Epidemiological Simulation**: SIR/SEIR disease spread models (coming soon)
- **Cellular Automaton Simulation**: Grid-based models with local update rules (coming soon)
- **Supply Chain Simulation**: Model multi-tier supply chains with inventory management (coming soon)

## Common Features

All SimLab simulators share these common features:

- **Consistent Interface**: All simulators inherit from BaseSimulation and provide a consistent API
- **Registry System**: Dynamic discovery and instantiation of simulation models
- **Parameter Validation**: Comprehensive input validation and error handling
- **Visualization Support**: Integration with common plotting libraries
- **Stochastic Processes**: Support for random processes with seed control for reproducibility
- **Extensibility**: Easy to extend with custom behavior

## Getting Started

To use any simulation in SimLab, follow this general pattern:

```python
from sim_lab.core import SimulatorRegistry

# Method 1: Create using the registry
sim = SimulatorRegistry.create(
    "SimulatorName",
    param1=value1,
    param2=value2
)

# Method 2: Create directly
from sim_lab.core import SpecificSimulation

sim = SpecificSimulation(
    param1=value1,
    param2=value2
)

# Run the simulation
results = sim.run_simulation()

# Analyze results
# (Each simulator provides specific methods for analysis)
```

Check the documentation for each specific simulator to learn about its parameters, methods, and examples.