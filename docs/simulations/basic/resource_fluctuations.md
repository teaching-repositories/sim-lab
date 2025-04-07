# Resource Fluctuations Simulation

The Resource Fluctuations Simulation models the price dynamics of commodities and resources over time, accounting for market volatility, trends, and supply disruptions.

## Overview

This simulation is designed to model resource pricing with the following key features:

- Daily price changes follow a normal distribution
- Adjustable volatility parameter to control price variation
- Market drift to model overall price trends
- Support for supply disruption events with customizable impact
- Reproducible results with random seed control

## Basic Usage

```python
from sim_lab.core import SimulatorRegistry

# Create a resource price simulation
sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=85.0,        # Starting price per unit
    days=365,                # Simulate for a year
    volatility=0.015,        # 1.5% daily volatility
    drift=0.0003,            # Slight upward trend
    random_seed=42           # For reproducible results
)

# Run the simulation
prices = sim.run_simulation()

# Access the results
print(f"Starting price: ${prices[0]:.2f} per unit")
print(f"Final price: ${prices[-1]:.2f} per unit")
print(f"Price change: {(prices[-1]/prices[0] - 1) * 100:.2f}%")
```

## Parameters

The Resource Fluctuations Simulation accepts the following parameters:

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `start_price` | float | The initial price of the resource | Yes | - |
| `days` | int | The duration of the simulation in days | Yes | - |
| `volatility` | float | The volatility of price changes | Yes | - |
| `drift` | float | The average daily price change trend | Yes | - |
| `supply_disruption_day` | int | The day a supply disruption occurs | No | None |
| `disruption_severity` | float | The magnitude of the disruption's impact | No | 0 |
| `random_seed` | int | Seed for random number generation | No | None |

## Understanding the Model

### Price Generation

The resource price for each day is calculated using the following formula:

```
price[day] = price[day-1] * (1 + random_change)
```

Where `random_change` is drawn from a normal distribution with mean `drift` and standard deviation `volatility`.

### Supply Disruptions

If a `supply_disruption_day` is specified, a supply disruption occurs on that day with the following impact:

```
price[disruption_day] = price[disruption_day-1] * (1 + disruption_severity)
```

This can model events like:
- Natural disasters affecting resource production
- Political events disrupting supply chains
- Production facility failures
- Trade restrictions or embargoes

## Example: Modeling a Supply Chain Disruption

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create a simulation with a major supply disruption
sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=85.0,
    days=365,
    volatility=0.01,
    drift=0.0002,
    supply_disruption_day=180,    # Disruption after 6 months
    disruption_severity=0.25,     # 25% price spike
    random_seed=42
)

# Run the simulation
prices = sim.run_simulation()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(prices)
plt.axvline(x=180, color='r', linestyle='--', label='Supply Disruption')
plt.title('Resource Price with Supply Chain Disruption')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Comparing Different Resources

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create simulations for different resource types
oil = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=75.0,
    days=365,
    volatility=0.018,      # High volatility
    drift=0.0004,          # Slight upward trend
    random_seed=42
)

natural_gas = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=4.5,
    days=365,
    volatility=0.022,      # Very high volatility
    drift=0.0003,          # Slight upward trend
    supply_disruption_day=240,  # Winter supply issues
    disruption_severity=0.15,   # 15% price spike
    random_seed=42
)

metals = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=2200.0,
    days=365,
    volatility=0.009,      # Lower volatility
    drift=0.0002,          # Slight upward trend
    random_seed=42
)

# Run simulations
oil_prices = oil.run_simulation()
gas_prices = natural_gas.run_simulation()
metal_prices = metals.run_simulation()

# Normalize prices for comparison (starting from 100)
norm_oil = [price / oil_prices[0] * 100 for price in oil_prices]
norm_gas = [price / gas_prices[0] * 100 for price in gas_prices]
norm_metal = [price / metal_prices[0] * 100 for price in metal_prices]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(norm_oil, label='Oil')
plt.plot(norm_gas, label='Natural Gas')
plt.plot(norm_metal, label='Industrial Metals')
plt.axvline(x=240, color='r', linestyle='--', alpha=0.5, label='Gas Supply Disruption')
plt.title('Normalized Resource Prices (Starting at 100)')
plt.xlabel('Days')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Seasonal Resource Pricing

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Create base simulation
days = 365 * 2  # Two years

sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=3.5,  # Starting price
    days=days,
    volatility=0.01,
    drift=0.0001,
    random_seed=42
)

# Run the baseline simulation
base_prices = sim.run_simulation()

# Add seasonal component
seasonal_prices = []
for day in range(days):
    # Add seasonal component (higher in winter, lower in summer)
    # Higher amplitude for natural gas pricing seasonality
    seasonal_factor = 0.15 * np.sin(2 * np.pi * (day % 365) / 365 + np.pi)  # Peak in winter
    seasonal_prices.append(base_prices[day] * (1 + seasonal_factor))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(base_prices, label='Base Model', alpha=0.5)
plt.plot(seasonal_prices, label='With Seasonality')
plt.title('Natural Gas Prices with Seasonal Component')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Extract winter and summer pricing
winter_prices = [seasonal_prices[i] for i in range(days) if (i % 365) in range(335, 365) or (i % 365) in range(0, 60)]
summer_prices = [seasonal_prices[i] for i in range(days) if (i % 365) in range(152, 243)]

print(f"Average winter price: ${np.mean(winter_prices):.2f}")
print(f"Average summer price: ${np.mean(summer_prices):.2f}")
print(f"Seasonal price differential: {(np.mean(winter_prices)/np.mean(summer_prices) - 1) * 100:.2f}%")
```

## Advanced Topics

### Modeling Multiple Supply Disruptions

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Base simulation
sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=85.0,
    days=730,  # 2 years
    volatility=0.012,
    drift=0.0002,
    random_seed=42
)

# Run the simulation to get baseline prices
base_prices = sim.run_simulation()

# Create a custom simulation with multiple disruptions
disruptions = [
    {"day": 90, "severity": 0.12},   # Small disruption
    {"day": 250, "severity": 0.25},  # Major disruption
    {"day": 400, "severity": -0.10}, # Supply increase (negative disruption)
    {"day": 600, "severity": 0.18}   # Another disruption
]

# Apply the disruptions
prices = [base_prices[0]]
for day in range(1, len(base_prices)):
    # Start with the base change
    new_price = base_prices[day]
    
    # Check if this day has a disruption
    for disruption in disruptions:
        if day == disruption["day"]:
            new_price = prices[-1] * (1 + disruption["severity"])
            break
            
    prices.append(new_price)

# Visualize
plt.figure(figsize=(14, 7))
plt.plot(base_prices, 'b--', alpha=0.5, label='Baseline')
plt.plot(prices, 'b-', label='With Disruptions')

# Mark disruptions
colors = ['r', 'darkred', 'g', 'orange']
for i, disruption in enumerate(disruptions):
    day = disruption["day"]
    severity = disruption["severity"]
    plt.axvline(x=day, color=colors[i], linestyle='--', alpha=0.7)
    direction = "increase" if severity > 0 else "decrease"
    plt.text(day+5, prices[day], f"{abs(severity)*100:.0f}% {direction}", 
             color=colors[i], fontweight='bold')

plt.title('Resource Price with Multiple Supply Disruptions')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

### Price Mean Reversion

In many resource markets, prices tend to revert to a long-term mean. You can model this:

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
import numpy as np
from typing import List, Optional

@SimulatorRegistry.register("MeanRevertingResource")
class MeanRevertingResourceSimulation(BaseSimulation):
    """Resource simulation with mean reversion."""
    
    def __init__(
        self,
        start_price: float,
        days: int,
        volatility: float,
        long_term_mean: float,
        reversion_rate: float,
        random_seed: Optional[int] = None
    ):
        """Initialize the mean-reverting resource simulation."""
        super().__init__(days=days, random_seed=random_seed)
        self.start_price = start_price
        self.volatility = volatility
        self.long_term_mean = long_term_mean
        self.reversion_rate = reversion_rate
    
    def run_simulation(self) -> List[float]:
        """Run the simulation with mean reversion."""
        self.reset()
        
        prices = [self.start_price]
        for day in range(1, self.days):
            previous_price = prices[-1]
            # Mean reversion component
            mean_reversion = self.reversion_rate * (self.long_term_mean - previous_price)
            # Random component
            random_change = np.random.normal(0, self.volatility)
            # New price
            new_price = previous_price * (1 + mean_reversion + random_change)
            prices.append(max(0.01, new_price))  # Ensure price doesn't go negative
            
        return prices

# Create and run the simulation
mean_reverting = MeanRevertingResourceSimulation(
    start_price=85.0,
    days=1000,
    volatility=0.02,
    long_term_mean=90.0,
    reversion_rate=0.02,  # Speed of reversion to the mean
    random_seed=42
)

prices = mean_reverting.run_simulation()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(prices)
plt.axhline(y=mean_reverting.long_term_mean, color='r', linestyle='--', 
           label=f'Long-term Mean (${mean_reverting.long_term_mean:.2f})')
plt.title('Mean-Reverting Resource Price')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

## API Reference

### ResourceFluctuationsSimulation

```python
ResourceFluctuationsSimulation(
    start_price: float,
    days: int,
    volatility: float,
    drift: float,
    supply_disruption_day: Optional[int] = None,
    disruption_severity: float = 0,
    random_seed: Optional[int] = None
)
```

#### Methods

- **run_simulation()**: Run the simulation and return a list of resource prices over time.
- **reset()**: Reset the simulation to its initial state.

## Further Reading

For more information about resource price modeling:

- [Commodity Price Dynamics](https://en.wikipedia.org/wiki/Commodity_markets)
- [Mean Reversion in Financial Markets](https://en.wikipedia.org/wiki/Mean_reversion_(finance))
- [Supply and Demand Shocks](https://en.wikipedia.org/wiki/Supply_shock)
- [Energy Price Forecasting](https://www.eia.gov/forecasts/)