from simnexus import ProductPopularitySimulation
import matplotlib.pyplot as plt

# Setting up a scenario with moderate natural growth and a significant marketing campaign.
sim = ProductPopularitySimulation(start_demand=500, days=180, growth_rate=0.02, 
                                  marketing_impact=0.1, promotion_day=30, promotion_effectiveness=0.5)

demand = sim.run_simulation()

# Visualizing product popularity
plt.figure(figsize=(10, 6))
plt.plot(demand, label='Product Demand')
plt.axvline(x=sim.promotion_day, color='blue', linestyle='--', label='Marketing Campaign Start')
plt.xlabel('Days')
plt.ylabel('Demand Units')
plt.title('Product Popularity Simulation')
plt.legend()
plt.show()