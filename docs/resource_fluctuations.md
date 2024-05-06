# Resource Fluctuation Simulation

## Purpose

This simulation models the price fluctuations of a critical resource for a business. It is applicable to commodities, raw materials, or any essential business input, helping students understand market dynamics and pricing strategies.

## Parameters

- `start_price`: The initial price of the resource.
- `days`: The duration of the simulation.
- `volatility`: Controls the severity of day-to-day price fluctuations.
- `drift`: Indicates the general upward or downward trend in price over time.
- `supply_disruption_day`: Specifies the day on which a supply disruption event occurs (optional).
- `disruption_severity`: The magnitude of the supply disruption's impact on the price, positive indicating a shortage and negative indicating a surplus.


**Example Code**

```python
from simulacra import ResourceFluctuationsSimulation
import matplotlib.pyplot as plt

# Setting up a moderate volatility and upward drift scenario with a supply disruption.
sim = ResourceFluctuationsSimulation(start_price=100, days=250, volatility=0.015, 
                         drift=0.0003, supply_disruption_day=100, disruption_severity=0.3) 

prices = sim.run_simulation()

# Visualising the price simulation
plt.figure(figsise=(10, 6))
plt.plot(prices, label='Resource Price')
plt.axvline(x=sim.supply_disruption_day, color='r', linestyle='--', label='Supply Disruption')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Resource Price Simulation')
plt.legend()
plt.show()
```
## Conducting Visual Analysis Using the Simulation:

Experiment! Use the simulation to explore and test various scenarios. Adjust parameters, try different strategies, and analyse the outcomes to gain deeper insights into resource management under fluctuating conditions.

- **Baseline Scenario Without Disruptions**: Begin by simulating the price path without any disruptions to establish a baseline for comparison with more complex scenarios.
  
- **Labeling and Annotations**: Ensure that your plots clearly show the days on the x-axis and price on the y-axis. Use lines or markers to indicate the day of a supply disruption or the implementation of a hedging strategy.

- **Interactive Exploration**: If tools are available, adjust parameters such as volatility and drift dynamically to observe how these changes affect the price simulation. This can help in understanding the immediate effects of each parameter.

- **Comparative Analysis**: Conduct side-by-side comparisons of scenarios with different levels of volatility or different strategies to visually assess their impact. This can make it easier to understand which conditions or strategies lead to the most favorable outcomes.  Consider calculating and comparing statistics such as the average price before and after a disruption event to quantify its impact.


## Task Specific Guidance

### Assess the Impact of Volatility on Price Stability:

Start by exploring how different levels of volatility affect day-to-day price fluctuations. This will help you understand the sensitivity of resource prices to changes in market conditions. Questions to Consider:

  - What trends do you notice as volatility increases? How does it affect the predictability of price movements?
  - How do different volatility levels impact the overall risk profile of investing in this resource?

### Model a Supply Disruption Event and Analyse Its Impact

Set up scenarios where a supply disruption occurs at a predetermined day. Change the severity of these disruptions to see how they influence resource prices. Questions to Consider:

  - How does the timing of a supply disruption affect its impact on resource prices?
  - Compare the prices before and after the disruption. What can you infer about the resilience of the market to sudden changes?

### (Optional) Explore Hedging Strategies

Implement simple hedging strategies to see how they could mitigate the risks associated with price volatility and supply disruptions. Consider strategies like futures contracts or options.Questions to Consider:

  - Which hedging strategy appears most effective in stabilising price fluctuations?
  - How do the costs of these strategies compare to their benefits in terms of reduced price volatility?

## Model Formulation

The formula used in the `ResourceFluctuationsSimulation` class is designed to simulate the fluctuations in resource prices, incorporating daily volatility, a trend or drift over time, and the effects of supply disruptions. Here's a breakdown of how the formula works for each day of the simulation:

1. **Volatility and Drift:** Each day, the price of the resource changes based on a combination of volatility and drift. The volatility represents the day-to-day variability in price changes, while the drift represents a consistent trend in price changes over time. This is modeled using a normal distribution where the mean of the distribution is given by the `drift` and the standard deviation by the `volatility`. This is expressed as:
   \[
   \text{Random Change} = \text{Normal}(\text{Drift}, \text{Volatility})
   \]
   Then, the new price is calculated as:
   \[
   \text{New Price} = \text{Previous Price} \times (1 + \text{Random Change})
   \]

2. **Supply Disruption:** If there's a day specified for a supply disruption (given by `supply_disruption_day`), the formula adjusts the price of the resource significantly based on the `disruption_severity`. The severity is modeled as a multiplicative factor to the price of the resource:
   \[
   \text{New Price} = \text{Previous Price} \times (1 + \text{Disruption Severity})
   \]

### Relation to Classical Models

The simulation model appears to draw from the classical geometric Brownian motion (GBM) model, which is commonly used in financial mathematics to model stock prices and other financial assets. In GBM, the logarithm of the stock prices follows a Brownian motion (also known as a Wiener process) with drift and volatility, similar to the structure used in this resource simulation class:

- **Geometric Brownian Motion:** The use of `previous_price * (1 + random_change)` closely resembles the discrete approximation of GBM, where price changes are log-normally distributed, allowing the price to stay positive and fluctuate in a realistic manner.

- **Supply Disruption as a Jump Process:** The inclusion of supply disruption as a multiplicative effect on the price for a specific day can be seen as a form of a jump process, where the price can have sudden, significant changes due to external events. This is similar to models used in energy markets and commodities trading, where sudden events can cause significant price changes.

Overall, while the exact parameters and implementation details might differ based on the simulation's objectives and the specific market being modeled, the underlying principles of the formula are well-established in the field of quantitative finance and economic modeling.