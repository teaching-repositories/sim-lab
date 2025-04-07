"""
Basic example of using SimLab to simulate a stock market scenario.

This example demonstrates:
1. Creating a StockMarketSimulation instance
2. Running the simulation
3. Visualizing the results with matplotlib
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Example scenario: High volatility with a downward price trend and a significant market event.
sim = SimulatorRegistry.create(
    "StockMarket",
    start_price=100,     # Starting price of the stock
    days=365,            # Simulate for one year
    volatility=0.03,     # Daily volatility of 3%
    drift=-0.001,        # Slight downward trend (-0.1% per day)
    event_day=100,       # Major market event on day 100
    event_impact=-0.2    # Event causes 20% drop in price
)

# Run the simulation and get results
prices = sim.run_simulation()

# Calculate some statistics
min_price = min(prices)
max_price = max(prices)
final_price = prices[-1]
change_pct = ((final_price - prices[0]) / prices[0]) * 100

# Print summary statistics
print(f"Simulation Summary:")
print(f"- Starting Price: ${prices[0]:.2f}")
print(f"- Final Price: ${final_price:.2f}")
print(f"- Change: {change_pct:.2f}%")
print(f"- Minimum Price: ${min_price:.2f}")
print(f"- Maximum Price: ${max_price:.2f}")
print(f"- Major Event: Day {sim.event_day}, Impact: {sim.event_impact * 100:.0f}%")

# Visualize the stock market fluctuations
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Stock Price')
plt.axvline(x=sim.event_day, color='red', linestyle='--', label='Major Market Event')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Stock Market Simulation')
plt.grid(True, alpha=0.3)
plt.legend()

# Save the figure (optional)
# plt.savefig("stock_market_simulation.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()