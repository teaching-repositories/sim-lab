from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt

# Setting up a moderate volatility and upward drift scenario with a supply disruption.
sim = SimulatorRegistry.create(
    "ResourceFluctuations",
    start_price=100, 
    days=250, 
    volatility=0.015, 
    drift=0.0003, 
    supply_disruption_day=100, 
    disruption_severity=0.3
) 

prices = sim.run_simulation()

# Visualising the price simulation
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Resource Price')
plt.axvline(x=sim.supply_disruption_day, color='r', linestyle='--', label='Supply Disruption')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Resource Price Simulation')
plt.legend()
plt.show()