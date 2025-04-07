# Educational Tools

SimLab is designed to be an effective educational tool for teaching simulation concepts, programming, and domain-specific modeling. This guide demonstrates how to use SimLab in educational settings.

## Overview

Simulations provide a powerful way to engage students in active learning about complex systems. SimLab offers several features that make it particularly well-suited for educational contexts:

- **Consistent API**: All simulators follow the same pattern, minimizing cognitive load
- **Multiple interfaces**: CLI, TUI, Web, and Python API for different learning styles
- **Visualization tools**: Built-in plotting capabilities for immediate visual feedback
- **Progressive complexity**: From basic to advanced simulation types
- **Registry system**: Demonstrates software design patterns

## Teaching Basic Simulation Concepts

### Example: Demonstrating Randomness and Seed Control

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create two simulations with different seeds
sim1 = SimulatorRegistry.create(
    "StockMarket",
    start_price=100,
    days=100,
    volatility=0.02,
    drift=0.001,
    random_seed=42  # Fixed seed
)

sim2 = SimulatorRegistry.create(
    "StockMarket",
    start_price=100,
    days=100,
    volatility=0.02,
    drift=0.001,
    random_seed=None  # Random seed
)

# Run the simulations multiple times
results1 = []
results2 = []

for i in range(3):
    # Run with fixed seed - should get identical results
    prices1 = sim1.run_simulation()
    results1.append(prices1)
    
    # Run with no fixed seed - should get different results each time
    prices2 = sim2.run_simulation()
    results2.append(prices2)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, result in enumerate(results1):
    ax1.plot(result, label=f'Run {i+1}')
ax1.set_title('Fixed Seed (42): Deterministic Results')
ax1.set_xlabel('Days')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True)

for i, result in enumerate(results2):
    ax2.plot(result, label=f'Run {i+1}')
ax2.set_title('No Fixed Seed: Random Results')
ax2.set_xlabel('Days')
ax2.set_ylabel('Price')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Teaching Programming Concepts

SimLab can be used to demonstrate various programming concepts:

### Object-Oriented Programming

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
import random
from typing import List

# Demonstrate subclassing and inheritance
@SimulatorRegistry.register("DiceRoll")
class DiceSimulation(BaseSimulation):
    """Simulates rolling dice."""
    
    def __init__(self, sides: int, num_dice: int, days: int, random_seed=None):
        """Initialize the simulation.
        
        Args:
            sides: Number of sides on each die
            num_dice: Number of dice to roll
            days: Number of days to simulate
            random_seed: Random seed for reproducibility
        """
        super().__init__(days=days, random_seed=random_seed)
        self.sides = sides
        self.num_dice = num_dice
    
    def run_simulation(self) -> List[int]:
        """Run the simulation.
        
        Returns:
            List of dice roll sums for each day
        """
        results = []
        for _ in range(self.days):
            # Roll the dice and sum the results
            roll_sum = sum(random.randint(1, self.sides) for _ in range(self.num_dice))
            results.append(roll_sum)
        return results

# Create the simulation
dice_sim = SimulatorRegistry.create(
    "DiceRoll",
    sides=6,
    num_dice=2,
    days=1000,
    random_seed=42
)

# Run the simulation and analyze results
results = dice_sim.run_simulation()

# Count frequency of each outcome
frequencies = {}
for roll in results:
    frequencies[roll] = frequencies.get(roll, 0) + 1

# Print results
for outcome in sorted(frequencies.keys()):
    print(f"Sum {outcome}: {frequencies[outcome]} times ({frequencies[outcome]/len(results)*100:.2f}%)")
```

## Course Integration Ideas

SimLab can be integrated into various courses:

1. **Computer Science**:
   - Data Structures and Algorithms
   - Object-Oriented Programming
   - Software Engineering

2. **Mathematics**:
   - Statistics and Probability
   - Differential Equations
   - Applied Mathematics

3. **Business**:
   - Finance and Investment
   - Marketing Analytics
   - Operations Research

4. **Science**:
   - Systems Biology
   - Population Ecology
   - Epidemiology

## Assignments and Projects

### Sample Assignment: Stock Market Analysis

**Objective**: Analyze how changing parameters affects stock market simulation

**Instructions**:
1. Create a stock market simulation with the following base parameters:
   - start_price: 100
   - days: 252 (one trading year)
   - volatility: 0.015
   - drift: 0.0005
   - random_seed: 42

2. Run the simulation and record the final price and overall return.

3. Vary each parameter individually and analyze how it affects the results:
   - Try volatility values: 0.005, 0.01, 0.02, 0.05
   - Try drift values: -0.001, 0, 0.001, 0.002
   - Try different random seeds

4. Create visualizations comparing the results.

5. Write a brief report explaining how each parameter affects the simulation outcomes.

## Further Resources

For more information on using SimLab for teaching, see:

- [Teaching Guide](../teaching_guide.md)
- [SimLab API Reference](../api.md)
- [Example Repository](https://github.com/teaching-repositories/sim-lab/tree/main/examples)