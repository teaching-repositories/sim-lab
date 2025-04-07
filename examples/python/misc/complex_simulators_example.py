"""
Example demonstrating the use of complex simulators.

This script shows how to use:
1. Agent-Based Simulation
2. System Dynamics Simulation
3. Supply Chain Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from sim_lab.core import (
    # Agent-based simulation
    AgentBasedSimulation, Agent, Environment,
    
    # System dynamics
    SystemDynamicsSimulation, Stock, Flow, Auxiliary, create_predefined_model,
    
    # Supply chain
    SupplyChainSimulation, SupplyChainNode, Factory, Distributor, Retailer, SupplyChainLink,
    base_stock_policy, seasonal_demand
)


def run_agent_based_example():
    """Run an agent-based simulation example (predator-prey model)."""
    print("\n--- Agent-Based Simulation Example: Predator-Prey Model ---")
    
    # Define our agent types
    class PreyAgent(Agent):
        """Represents a prey animal in the simulation."""
        
        def __init__(self, agent_id, position=None):
            """Initialize a prey agent."""
            initial_state = {
                "type": "prey",
                "energy": 20,
                "age": 0,
                "reproduce_countdown": np.random.randint(5, 10)
            }
            super().__init__(agent_id, initial_state, position or (np.random.random() * 100, np.random.random() * 100))
        
        def update(self, environment, neighbors):
            """Update the prey's state."""
            # Age the prey
            self.state["age"] += 1
            
            # Lose energy
            self.state["energy"] -= 1
            
            # Reproduce if ready
            self.state["reproduce_countdown"] -= 1
            
            # Random movement (simple for this example)
            dx = np.random.uniform(-5, 5)
            dy = np.random.uniform(-5, 5)
            
            x, y = self.position
            x = max(0, min(100, x + dx))
            y = max(0, min(100, y + dy))
            self.position = (x, y)
            
            # Find nearby predators and try to move away
            predators = [n for n in neighbors if n.state["type"] == "predator"]
            if predators:
                # Simple avoidance behavior
                closest = predators[0]
                dx = self.position[0] - closest.position[0]
                dy = self.position[1] - closest.position[1]
                dist = max(0.1, (dx**2 + dy**2)**0.5)
                
                # Move away from predator
                x, y = self.position
                x = max(0, min(100, x + dx/dist * 3))
                y = max(0, min(100, y + dy/dist * 3))
                self.position = (x, y)
            
            # Find food in environment
            if environment.state.get("grass", 0) > 0:
                self.state["energy"] += 5
                environment.state["grass"] -= 1
    
    class PredatorAgent(Agent):
        """Represents a predator animal in the simulation."""
        
        def __init__(self, agent_id, position=None):
            """Initialize a predator agent."""
            initial_state = {
                "type": "predator",
                "energy": 30,
                "age": 0,
                "reproduce_countdown": np.random.randint(10, 15)
            }
            super().__init__(agent_id, initial_state, position or (np.random.random() * 100, np.random.random() * 100))
        
        def update(self, environment, neighbors):
            """Update the predator's state."""
            # Age the predator
            self.state["age"] += 1
            
            # Lose energy
            self.state["energy"] -= 2
            
            # Reproduce if ready
            self.state["reproduce_countdown"] -= 1
            
            # Random movement (simple for this example)
            dx = np.random.uniform(-3, 3)
            dy = np.random.uniform(-3, 3)
            
            x, y = self.position
            x = max(0, min(100, x + dx))
            y = max(0, min(100, y + dy))
            self.position = (x, y)
            
            # Find nearby prey and try to eat them
            prey = [n for n in neighbors if n.state["type"] == "prey"]
            if prey and self.state["energy"] < 50:
                # Chase and eat prey
                closest = prey[0]
                dx = closest.position[0] - self.position[0]
                dy = closest.position[1] - self.position[1]
                dist = max(0.1, (dx**2 + dy**2)**0.5)
                
                # Move toward prey
                x, y = self.position
                x = max(0, min(100, x + dx/dist * 5))
                y = max(0, min(100, y + dy/dist * 5))
                self.position = (x, y)
                
                # If close enough, eat prey
                if dist < 5:
                    self.state["energy"] += 15
                    environment.state["prey_eaten"] = environment.state.get("prey_eaten", 0) + 1
                    closest.state["energy"] = 0  # Mark for removal
    
    # Create a custom environment update method
    class PredatorPreyEnvironment(Environment):
        """Environment for the predator-prey simulation."""
        
        def __init__(self):
            """Initialize the environment."""
            initial_state = {
                "grass": 100,
                "prey_eaten": 0,
                "day": 0
            }
            super().__init__(initial_state)
        
        def update(self, agents):
            """Update the environment based on agent states."""
            # Grow more grass
            self.state["grass"] = min(200, self.state["grass"] + 5)
            self.state["day"] += 1
    
    # Create the agent factory
    def agent_factory(agent_id):
        """Create agents, alternating between prey and predators, with more prey."""
        if agent_id % 4 == 0:
            return PredatorAgent(agent_id)
        else:
            return PreyAgent(agent_id)
    
    # Create the environment
    environment = PredatorPreyEnvironment()
    
    # Create the simulation
    sim = AgentBasedSimulation(
        agent_factory=agent_factory,
        num_agents=100,  # Start with 100 agents
        environment=environment,
        days=100,
        neighborhood_radius=10.0,
        save_history=True,
        random_seed=42
    )
    
    # Run the simulation
    metrics = sim.run_simulation()
    
    # Process simulation results
    predator_counts = []
    prey_counts = []
    
    for m in metrics:
        # Count agents by type
        predator_count = m.get("type_predator", 0)
        prey_count = m.get("type_prey", 0)
        
        predator_counts.append(predator_count)
        prey_counts.append(prey_count)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(predator_counts, label='Predators')
    plt.plot(prey_counts, label='Prey')
    plt.title('Predator-Prey Population Dynamics')
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.savefig("predator_prey.png")
    print("Saved Predator-Prey results to 'predator_prey.png'")


def run_system_dynamics_example():
    """Run a system dynamics simulation example (predator-prey model using Lotka-Volterra equations)."""
    print("\n--- System Dynamics Simulation Example: Lotka-Volterra Model ---")
    
    # Use the predefined Lotka-Volterra model
    lotka_volterra_sim = create_predefined_model(
        'lotka_volterra',
        prey_growth_rate=0.1,      # Growth rate of prey population
        predation_rate=0.01,       # Rate at which predators consume prey
        predator_growth_rate=0.005, # Rate at which predators convert prey into new predators
        predator_death_rate=0.05,  # Natural death rate of predators
        initial_prey=100,          # Initial prey population
        initial_predators=20,      # Initial predator population
        days=200,                  # Simulation duration
        dt=0.1                     # Time step size
    )
    
    # Run the simulation
    results = lotka_volterra_sim.run_simulation()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['stock_prey'], label='Prey')
    plt.plot(results['stock_predators'], label='Predators')
    plt.title('Lotka-Volterra Predator-Prey Model')
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.savefig("lotka_volterra.png")
    print("Saved Lotka-Volterra results to 'lotka_volterra.png'")
    
    # Create a custom system dynamics model: SIR epidemic model
    print("\n--- System Dynamics Simulation Example: SIR Epidemic Model ---")
    
    # Use the predefined SIR model
    sir_sim = create_predefined_model(
        'sir',
        population=10000,
        initial_infected=10,
        initial_recovered=0,
        transmission_rate=0.3,
        recovery_rate=0.1,
        days=100,
        dt=0.1
    )
    
    # Run the simulation
    results = sir_sim.run_simulation()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['stock_susceptible'], label='Susceptible')
    plt.plot(results['stock_infected'], label='Infected')
    plt.plot(results['stock_recovered'], label='Recovered')
    plt.title('SIR Epidemic Model')
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.savefig("sir_system_dynamics.png")
    print("Saved SIR System Dynamics results to 'sir_system_dynamics.png'")


def run_supply_chain_example():
    """Run a supply chain simulation example (simple manufacturing supply chain)."""
    print("\n--- Supply Chain Simulation Example: Manufacturing Supply Chain ---")
    
    # Create nodes
    factory = Factory(
        name="Factory",
        production_capacity=100,
        production_cost=1.0,
        initial_inventory=200,
        capacity=500,
        lead_time=2
    )
    
    distributor = Distributor(
        name="Distributor",
        shipping_cost=0.5,
        initial_inventory=150,
        capacity=300,
        lead_time=3
    )
    
    retailer = Retailer(
        name="Retailer",
        selling_price=5.0,
        holding_cost=0.1,
        stockout_cost=1.0,
        initial_inventory=100,
        capacity=200,
        lead_time=1
    )
    
    # Create links
    factory_to_distributor = SupplyChainLink(factory, distributor)
    distributor_to_retailer = SupplyChainLink(distributor, retailer)
    
    # Create node dictionary
    nodes = {
        "Factory": factory,
        "Distributor": distributor,
        "Retailer": retailer
    }
    
    # Create links list
    links = [factory_to_distributor, distributor_to_retailer]
    
    # Create demand generator (seasonal demand)
    demand_gen = seasonal_demand(base_rate=50, amplitude=0.3, period=30)
    
    # Create ordering policies
    ordering_policies = {
        "Factory": lambda node, day, sim: {},  # Factory doesn't order from anyone
        "Distributor": base_stock_policy(target_level=200, review_period=3),
        "Retailer": base_stock_policy(target_level=150, review_period=2)
    }
    
    # Create the simulation
    supply_chain_sim = SupplyChainSimulation(
        nodes=nodes,
        links=links,
        demand_generator=demand_gen,
        ordering_policies=ordering_policies,
        days=100,
        random_seed=42
    )
    
    # Run the simulation
    results = supply_chain_sim.run_simulation()
    
    # Plot inventory levels
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(results['Factory']['inventory'], label='Factory Inventory')
    plt.title('Factory Inventory')
    plt.ylabel('Units')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(results['Distributor']['inventory'], label='Distributor Inventory')
    plt.title('Distributor Inventory')
    plt.ylabel('Units')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(results['Retailer']['inventory'], label='Retailer Inventory')
    plt.plot(results['Retailer']['demand'], 'r--', label='Customer Demand')
    plt.title('Retailer Inventory and Demand')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("supply_chain_inventory.png")
    print("Saved Supply Chain Inventory results to 'supply_chain_inventory.png'")
    
    # Plot financial metrics
    plt.figure(figsize=(10, 6))
    
    # Calculate cumulative metrics
    cumulative_revenue = np.cumsum(results['Retailer']['revenue'])
    
    factory_costs = np.cumsum(results['Factory']['production_cost'])
    shipping_costs = np.cumsum(results['Distributor']['shipping_cost'])
    holding_costs = np.cumsum(results['Retailer']['holding_cost'])
    stockout_costs = np.cumsum(results['Retailer']['stockout_cost'])
    
    cumulative_costs = factory_costs + shipping_costs + holding_costs + stockout_costs
    cumulative_profit = cumulative_revenue - cumulative_costs
    
    plt.plot(cumulative_revenue, label='Revenue')
    plt.plot(cumulative_costs, label='Costs')
    plt.plot(cumulative_profit, label='Profit')
    plt.title('Supply Chain Financial Metrics')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("supply_chain_financials.png")
    print("Saved Supply Chain Financial results to 'supply_chain_financials.png'")
    
    # Print overall metrics
    service_level = results['overall_metrics']['service_level'][0]
    total_profit = results['overall_metrics']['total_profit'][0]
    print(f"Supply Chain Service Level: {service_level:.2%}")
    print(f"Supply Chain Total Profit: ${total_profit:.2f}")


def main():
    """Run all complex simulator examples."""
    print("Running examples for complex simulators...\n")
    
    run_agent_based_example()
    run_system_dynamics_example()
    run_supply_chain_example()


if __name__ == "__main__":
    main()