# Scientific Simulation

SimLab provides tools for scientific simulations across various domains, from ecology to epidemiology. This guide demonstrates how to use SimLab for scientific research and modeling.

## Overview

Scientific simulation involves creating computational models of physical, biological, or social systems to study their behavior under different conditions. SimLab offers several simulation types specifically designed for scientific applications:

- **Agent-Based Simulation**: Model complex systems through autonomous agent interactions
- **Predator-Prey Simulation**: Model ecological population dynamics
- **Epidemiological Simulation**: Model disease spread using SIR/SEIR models
- **Network Simulation**: Model processes on complex networks
- **Markov Chain Simulation**: Model stochastic processes with the Markov property

## Example: Predator-Prey Dynamics

Here's an example of using SimLab to model predator-prey population dynamics:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create a predator-prey simulation
ecosystem = SimulatorRegistry.create(
    "PredatorPrey",
    prey_growth_rate=0.1,      # Prey growth rate
    predation_rate=0.02,       # Rate at which predators consume prey
    predator_death_rate=0.1,   # Natural death rate of predators
    predator_efficiency=0.2,   # Efficiency of converting prey to predator population
    initial_prey=100,
    initial_predators=15,
    days=500,
    random_seed=42
)

# Run the simulation
results = ecosystem.run_simulation()

# Extract population data
prey_pop = [day[0] for day in results]
predator_pop = [day[1] for day in results]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(prey_pop, 'g-', label='Prey')
plt.plot(predator_pop, 'r-', label='Predators')
plt.title('Predator-Prey Population Dynamics')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Epidemic Modeling

SimLab can model disease spread within a population:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create a SIR epidemiological model
epidemic = SimulatorRegistry.create(
    "Epidemiological",
    model_type="SIR",
    population=10000,
    initial_infected=10,
    transmission_rate=0.3,
    recovery_rate=0.1,
    days=100,
    random_seed=42
)

# Run the simulation
results = epidemic.run_simulation()

# Extract population compartments
susceptible = [day[0] for day in results]
infected = [day[1] for day in results]
recovered = [day[2] for day in results]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(susceptible, 'b-', label='Susceptible')
plt.plot(infected, 'r-', label='Infected')
plt.plot(recovered, 'g-', label='Recovered')
plt.title('SIR Epidemic Model')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
```

## Applications in Scientific Research

SimLab supports various scientific research applications:

1. **Ecology**: Model species interactions and population dynamics
2. **Epidemiology**: Study disease spread and intervention strategies
3. **Physics**: Simulate particle interactions and system behaviors
4. **Social Sciences**: Model social dynamics and network effects
5. **Climate Science**: Simulate resource fluctuations and environmental impacts

## Monte Carlo Methods

SimLab's Monte Carlo simulation capabilities are particularly useful for scientific applications:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Create a Monte Carlo simulation to estimate π
mc_sim = SimulatorRegistry.create(
    "MonteCarlo",
    dimensions=2,
    samples=10000,
    random_seed=42
)

# Define a function to check if a point is inside a unit circle
def point_in_circle(point):
    x, y = point
    return x**2 + y**2 <= 1

# Run the simulation
result = mc_sim.run_simulation(point_in_circle)

# Calculate π estimate (area of circle = π * r² where r=1, so π = 4 * fraction of points in circle)
pi_estimate = 4 * result["result"]

print(f"π estimate: {pi_estimate}")
print(f"Actual π: {np.pi}")
print(f"Error: {abs(pi_estimate - np.pi) / np.pi * 100:.6f}%")
```

## Further Resources

For more specific applications, see:

- [Agent-Based Simulation](../simulations/agent_based/agent_based.md)
- [Network Simulation](../simulations/network/network.md)
- [Monte Carlo Simulation](../simulations/statistical/monte_carlo.md)