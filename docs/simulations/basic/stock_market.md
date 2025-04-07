# Stock Market Simulation

The Stock Market Simulation models the fluctuations of stock prices over time, accounting for volatility, general market trends (drift), and specific market events.

## Overview

The stock market simulation is based on a modified random walk model with the following key features:

- Daily price changes follow a normal distribution
- Adjustable volatility to control price variation
- Market drift parameter to model bullish or bearish trends
- Support for major market events on specific days
- Reproducible results with random seed control

## Basic Usage

```python
from sim_lab.core import SimulatorRegistry

# Create a stock market simulation
sim = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,           # Typical number of trading days in a year
    volatility=0.02,    # 2% daily volatility
    drift=0.0005,       # 0.05% average daily growth
    random_seed=42      # For reproducible results
)

# Run the simulation
prices = sim.run_simulation()

# Access the results
print(f"Starting price: ${prices[0]:.2f}")
print(f"Final price: ${prices[-1]:.2f}")
print(f"Return: {(prices[-1]/prices[0] - 1) * 100:.2f}%")
```

## Parameters

The Stock Market Simulation accepts the following parameters:

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `start_price` | float | The initial price of the stock | Yes | - |
| `days` | int | The duration of the simulation in days | Yes | - |
| `volatility` | float | The volatility of stock price changes | Yes | - |
| `drift` | float | The average daily price change trend | Yes | - |
| `event_day` | int | The day a major market event occurs | No | None |
| `event_impact` | float | The magnitude of the event's impact | No | 0 |
| `random_seed` | int | Seed for random number generation | No | None |

## Understanding the Model

### Price Generation

The stock price for each day is calculated using the following formula:

```
price[day] = price[day-1] * (1 + random_change)
```

Where `random_change` is drawn from a normal distribution with mean `drift` and standard deviation `volatility`.

### Market Events

If an `event_day` is specified, a market event occurs on that day with the following impact:

```
price[event_day] = price[event_day-1] * (1 + event_impact)
```

This can model significant events like earnings announcements, market crashes, or other macroeconomic events.

## Example: Simulating a Market Crash

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create a simulation with a market crash
sim = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.015,
    drift=0.0008,
    event_day=125,       # Crash happens on day 125
    event_impact=-0.15,  # 15% market drop
    random_seed=42
)

# Run the simulation
prices = sim.run_simulation()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(prices)
plt.axvline(x=125, color='r', linestyle='--', label='Market Crash')
plt.title('Stock Price with Market Crash')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Comparing Market Conditions

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Create simulations for different market conditions
bull_market = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.01,
    drift=0.001,      # Strong positive drift (bull market)
    random_seed=42
)

bear_market = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.02,
    drift=-0.0005,    # Negative drift (bear market)
    random_seed=42
)

volatile_market = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.03,  # High volatility
    drift=0.0003,
    random_seed=42
)

# Run simulations
bull_prices = bull_market.run_simulation()
bear_prices = bear_market.run_simulation()
volatile_prices = volatile_market.run_simulation()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(bull_prices, label='Bull Market')
plt.plot(bear_prices, label='Bear Market')
plt.plot(volatile_prices, label='Volatile Market')
plt.title('Stock Prices Under Different Market Conditions')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

## Monte Carlo Analysis

You can run multiple simulations to assess the range of possible outcomes:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Parameters for all simulations
start_price = 100.0
days = 252
volatility = 0.02
drift = 0.0005

# Run multiple simulations
num_simulations = 100
all_prices = []

for i in range(num_simulations):
    sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        random_seed=None  # Different seed each time
    )
    prices = sim.run_simulation()
    all_prices.append(prices)

# Convert to numpy array for easier analysis
all_prices = np.array(all_prices)

# Calculate statistics
mean_price = np.mean(all_prices, axis=0)
median_price = np.median(all_prices, axis=0)
q5_price = np.percentile(all_prices, 5, axis=0)
q95_price = np.percentile(all_prices, 95, axis=0)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(mean_price, 'b', label='Mean Price')
plt.plot(median_price, 'g', label='Median Price')
plt.fill_between(range(days), q5_price, q95_price, color='b', alpha=0.2, label='90% Confidence Interval')
plt.title('Monte Carlo Simulation of Stock Prices')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Display final price statistics
final_prices = all_prices[:, -1]
print(f"Expected final price: ${np.mean(final_prices):.2f}")
print(f"Median final price: ${np.median(final_prices):.2f}")
print(f"5th percentile: ${np.percentile(final_prices, 5):.2f}")
print(f"95th percentile: ${np.percentile(final_prices, 95):.2f}")
```

## Advanced Topics

### Incorporating Dividend Payments

You can extend the model to incorporate dividend payments:

```python
from sim_lab.core import SimulatorRegistry

# Create a simulation with quarterly dividends
sim = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.015,
    drift=0.0007,
    random_seed=42
)

# Run the simulation
prices = sim.run_simulation()

# Manually add quarterly dividends (assume 1% dividend yield per quarter)
quarterly_dividend = 0.01 * sim.start_price / 4
dividend_days = [63, 126, 189, 252]  # Quarterly dividend days
total_dividends = 0

for day in dividend_days:
    if day < len(prices):
        total_dividends += quarterly_dividend

# Calculate total return including dividends
price_return = (prices[-1] / prices[0] - 1) * 100
dividend_return = (total_dividends / prices[0]) * 100
total_return = price_return + dividend_return

print(f"Price return: {price_return:.2f}%")
print(f"Dividend return: {dividend_return:.2f}%")
print(f"Total return: {total_return:.2f}%")
```

### Correlation Between Multiple Stocks

You can model correlations between multiple stocks:

```python
from sim_lab.core import SimulatorRegistry
import numpy as np
import matplotlib.pyplot as plt

# Number of stocks to simulate
num_stocks = 3
days = 252

# Create correlated random changes
# Use a correlation matrix
correlation_matrix = np.array([
    [1.0, 0.7, 0.3],  # Stock 1 correlations
    [0.7, 1.0, 0.5],  # Stock 2 correlations
    [0.3, 0.5, 1.0]   # Stock 3 correlations
])

# Parameters for each stock
start_prices = [100.0, 50.0, 75.0]
volatilities = [0.01, 0.015, 0.02]
drifts = [0.0005, 0.0007, 0.0003]

# Generate correlated normal random values
np.random.seed(42)
random_changes = np.random.multivariate_normal(
    mean=drifts,
    cov=np.outer(volatilities, volatilities) * correlation_matrix,
    size=days
)

# Initialize prices array
prices = np.zeros((num_stocks, days))
prices[:, 0] = start_prices

# Generate price paths
for day in range(1, days):
    for stock in range(num_stocks):
        prices[stock, day] = prices[stock, day-1] * (1 + random_changes[day-1, stock])

# Plot results
plt.figure(figsize=(12, 6))
for stock in range(num_stocks):
    plt.plot(prices[stock, :], label=f'Stock {stock+1}')

plt.title('Correlated Stock Price Movements')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate correlation of returns
returns = np.diff(prices, axis=1) / prices[:, :-1]
empirical_correlation = np.corrcoef(returns)
print("Empirical correlation matrix of returns:")
print(empirical_correlation)
```

## API Reference

### StockMarketSimulation

```python
StockMarketSimulation(
    start_price: float,
    days: int,
    volatility: float,
    drift: float,
    event_day: Optional[int] = None,
    event_impact: float = 0,
    random_seed: Optional[int] = None
)
```

#### Methods

- **run_simulation()**: Run the simulation and return a list of stock prices over time.
- **reset()**: Reset the simulation to its initial state.

## Further Reading

For more information about stock market modeling and financial simulations:

- [Random Walk Hypothesis](https://en.wikipedia.org/wiki/Random_walk_hypothesis)
- [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- [Option Pricing Models](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Market Volatility](https://en.wikipedia.org/wiki/Volatility_(finance))