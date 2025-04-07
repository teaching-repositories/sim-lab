# Business Modeling

SimLab provides powerful tools for modeling business scenarios and market dynamics. This guide demonstrates how to use SimLab for various business modeling applications.

## Overview

Business modeling involves simulating market conditions, product lifecycles, resource allocation, and other business processes to gain insights and support decision-making. SimLab offers several simulators specifically designed for business applications:

- **Stock Market Simulation**: Model stock price fluctuations and market events
- **Resource Fluctuations Simulation**: Model commodity price dynamics and supply disruptions
- **Product Popularity Simulation**: Model product demand trends and marketing impacts
- **Supply Chain Simulation**: Model multi-tier supply chains with inventory management

## Example: Market Analysis

Here's an example of using SimLab to analyze different market scenarios:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Create simulations for different market conditions
bull_market = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,  # Trading days in a year
    volatility=0.01,
    drift=0.001,  # Strong positive drift
    random_seed=42
)

bear_market = SimulatorRegistry.create(
    "StockMarket",
    start_price=100.0,
    days=252,
    volatility=0.02,
    drift=-0.0005,  # Negative drift
    random_seed=42
)

# Run simulations
bull_prices = bull_market.run_simulation()
bear_prices = bear_market.run_simulation()

# Calculate end-of-year returns
bull_return = (bull_prices[-1] / bull_prices[0] - 1) * 100
bear_return = (bear_prices[-1] / bear_prices[0] - 1) * 100

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(bull_prices, 'g-', label=f'Bull Market (+{bull_return:.1f}%)')
plt.plot(bear_prices, 'r-', label=f'Bear Market ({bear_return:.1f}%)')
plt.title('Stock Price Scenarios')
plt.xlabel('Trading Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
```

## Example: Product Launch Strategy

SimLab can help assess different product launch strategies:

```python
from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Strategy 1: High initial marketing with promotion
strategy1 = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=50,
    days=365,
    growth_rate=0.003,
    marketing_impact=0.005,  # High marketing spend
    promotion_day=30,        # Launch promotion after 1 month
    promotion_effectiveness=0.3,
    random_seed=42
)

# Strategy 2: Gradual growth with sustained marketing
strategy2 = SimulatorRegistry.create(
    "ProductPopularity",
    start_demand=50,
    days=365,
    growth_rate=0.005,      # Higher natural growth
    marketing_impact=0.002, # Lower marketing spend
    random_seed=42
)

# Run simulations
demand1 = strategy1.run_simulation()
demand2 = strategy2.run_simulation()

# Calculate total sales for each strategy
total_sales1 = sum(demand1)
total_sales2 = sum(demand2)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(demand1, 'b-', label=f'High Marketing + Promo (Total: {total_sales1:.0f})')
plt.plot(demand2, 'g-', label=f'Organic Growth (Total: {total_sales2:.0f})')
plt.axvline(x=30, color='r', linestyle='--', label='Promotion')
plt.title('Product Launch Strategies: Demand Comparison')
plt.xlabel('Days')
plt.ylabel('Daily Demand (units)')
plt.legend()
plt.grid(True)
plt.show()
```

## Applications in Business Education

SimLab is particularly useful in business education contexts:

1. **MBA Programs**: Students can explore market dynamics and test business strategies
2. **Finance Courses**: Simulate trading strategies and market conditions
3. **Marketing Education**: Model product lifecycle and marketing campaign impacts
4. **Supply Chain Management**: Simulate inventory policies and supply disruptions

## Further Resources

For more specific applications, see:

- [Stock Market Simulation](../simulations/basic/stock_market.md)
- [Product Popularity Simulation](../simulations/basic/product_popularity.md)
- [Resource Fluctuations Simulation](../simulations/basic/resource_fluctuations.md)