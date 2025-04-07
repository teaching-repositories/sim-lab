"""
Basic example of using SimLab to simulate resource price fluctuations.

This example demonstrates:
1. Creating a ResourceFluctuationsSimulation instance
2. Running the simulation
3. Visualizing the results with matplotlib
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Setting up a moderate volatility and upward drift scenario with a supply disruption.
sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=100,           # Starting price of the resource
    days=250,                  # Simulate for 250 days
    volatility=0.015,          # Daily volatility of 1.5%
    drift=0.0003,              # Slight upward trend (0.03% per day)
    supply_disruption_day=100, # Supply disruption on day 100
    disruption_severity=0.3    # Disruption causes 30% increase in price
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
print(f"- Supply Disruption: Day {sim.supply_disruption_day}, Severity: {sim.disruption_severity * 100:.0f}%")

# Visualize the resource price fluctuations
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Resource Price')
plt.axvline(x=sim.supply_disruption_day, color='red', linestyle='--', label='Supply Disruption')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Resource Price Simulation')
plt.grid(True, alpha=0.3)
plt.legend()

# Save the figure (optional)
# plt.savefig("resource_fluctuations_simulation.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()