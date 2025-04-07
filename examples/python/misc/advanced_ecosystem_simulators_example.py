"""
Example demonstrating the use of advanced ecosystem simulators.

This script shows how to use:
1. Network/Graph Simulation
2. Markov Chain Simulation
3. Predator-Prey Ecosystem Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from sim_lab.core import (
    # Network simulation
    NetworkSimulation, create_random_network, create_scale_free_network, create_small_world_network,
    
    # Markov Chain simulation
    MarkovChainSimulation, create_weather_model, create_random_walk,
    
    # Predator-Prey simulation
    PredatorPreySimulation, create_predator_prey_model
)


def run_network_example():
    """Run a network simulation example."""
    print("\n--- Network Simulation Example ---")
    
    # Create different types of networks
    print("Creating three network types: Random, Scale-Free, and Small-World")
    
    # 1. Random network (Erdős–Rényi model)
    random_net = create_random_network(
        num_nodes=50,
        edge_probability=0.1,
        directed=False
    )
    
    # 2. Scale-free network (Barabási–Albert model)
    scale_free_net = create_scale_free_network(
        num_nodes=50,
        m=2,  # Each new node attaches to 2 existing nodes
        directed=False
    )
    
    # 3. Small-world network (Watts–Strogatz model)
    small_world_net = create_small_world_network(
        num_nodes=50,
        k=4,  # Each node connects to k nearest neighbors
        beta=0.1,  # Rewiring probability
        directed=False
    )
    
    # Define a network diffusion process
    def diffusion_update(network, day):
        """Update function for diffusion process on a network."""
        # Probability of transmission across an edge
        transmission_prob = 0.2
        
        # On day 1, seed the network with a few "infected" nodes
        if day == 1:
            for node_id in range(3):  # First 3 nodes are initially infected
                network.nodes[node_id].update_attribute("infected", True)
                network.nodes[node_id].update_attribute("infection_time", day)
        
        # For all other days, spread the infection
        for node_id, node in network.nodes.items():
            # Skip nodes that are already infected
            if node.attributes.get("infected", False):
                continue
            
            # Check neighbors for infection
            infected_neighbors = 0
            for neighbor_id in node.neighbors:
                if network.nodes[neighbor_id].attributes.get("infected", False):
                    infected_neighbors += 1
            
            # Probability of infection increases with more infected neighbors
            infection_prob = 1 - (1 - transmission_prob) ** infected_neighbors
            
            # Determine if node becomes infected
            if np.random.random() < infection_prob:
                node.update_attribute("infected", True)
                node.update_attribute("infection_time", day)
    
    # Add same diffusion process to all networks
    random_net.update_function = diffusion_update
    scale_free_net.update_function = diffusion_update
    small_world_net.update_function = diffusion_update
    
    # Run the simulations
    print("Running diffusion process on the networks...")
    random_metrics = random_net.run_simulation()
    scale_free_metrics = scale_free_net.run_simulation()
    small_world_metrics = small_world_net.run_simulation()
    
    # Calculate infection spread for each network
    def count_infected(network, day):
        return sum(1 for node in network.nodes.values() 
                 if node.attributes.get("infected", False) and 
                    node.attributes.get("infection_time", float('inf')) <= day)
    
    random_infected = [count_infected(random_net, day) for day in range(random_net.days)]
    scale_free_infected = [count_infected(scale_free_net, day) for day in range(scale_free_net.days)]
    small_world_infected = [count_infected(small_world_net, day) for day in range(small_world_net.days)]
    
    # Plot infection spread
    plt.figure(figsize=(10, 6))
    plt.plot(random_infected, label='Random Network')
    plt.plot(scale_free_infected, label='Scale-Free Network')
    plt.plot(small_world_infected, label='Small-World Network')
    plt.title('Infection Spread in Different Network Types')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Infected Nodes')
    plt.legend()
    plt.grid(True)
    plt.savefig("network_diffusion.png")
    print("Saved Network Diffusion results to 'network_diffusion.png'")
    
    # Compute and display final infection rates
    random_rate = random_infected[-1] / len(random_net.nodes)
    scale_free_rate = scale_free_infected[-1] / len(scale_free_net.nodes)
    small_world_rate = small_world_infected[-1] / len(small_world_net.nodes)
    
    print(f"Final infection rates:")
    print(f"  Random Network: {random_rate:.2%}")
    print(f"  Scale-Free Network: {scale_free_rate:.2%}")
    print(f"  Small-World Network: {small_world_rate:.2%}")


def run_markov_chain_example():
    """Run a Markov chain simulation example."""
    print("\n--- Markov Chain Simulation Example ---")
    
    # 1. Weather model
    print("Creating weather model...")
    weather_sim = create_weather_model(
        sunny_to_sunny=0.7,
        sunny_to_cloudy=0.2,
        sunny_to_rainy=0.1,
        cloudy_to_sunny=0.3,
        cloudy_to_cloudy=0.4,
        cloudy_to_rainy=0.3,
        rainy_to_sunny=0.2,
        rainy_to_cloudy=0.3,
        rainy_to_rainy=0.5,
        initial_state="Sunny",
        days=100
    )
    
    # Run the simulation
    weather_states = weather_sim.run_simulation()
    weather_names = weather_sim.get_state_names()
    
    # Calculate statistics
    state_dist = weather_sim.get_state_distribution()
    print(f"Weather state distribution:")
    for state, freq in state_dist.items():
        print(f"  {state}: {freq:.2%}")
    
    # Try to compute stationary distribution
    try:
        stationary_dist = weather_sim.compute_stationary_distribution()
        print(f"Stationary distribution:")
        for i, prob in enumerate(stationary_dist):
            print(f"  {weather_sim.states[i]}: {prob:.2%}")
    except ValueError as e:
        print(f"Could not compute stationary distribution: {e}")
    
    # 2. Random walk model
    print("\nCreating random walk model...")
    walk_sim = create_random_walk(
        p_up=0.6,
        p_down=0.4,
        initial_position=0,
        min_position=-10,
        max_position=10,
        days=200
    )
    
    # Run the simulation
    walk_states = walk_sim.run_simulation()
    walk_positions = [walk_sim.states[i] for i in walk_states]
    
    # Plot the weather model results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    weather_colors = {'Sunny': 'yellow', 'Cloudy': 'gray', 'Rainy': 'blue'}
    for day, state in enumerate(weather_names):
        plt.axvspan(day, day + 1, facecolor=weather_colors[state], alpha=0.5)
    plt.yticks([])
    plt.title('Weather Simulation')
    plt.xlabel('Day')
    
    plt.subplot(2, 1, 2)
    plt.plot(walk_positions)
    plt.title('Random Walk Simulation')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("markov_chain_results.png")
    print("Saved Markov Chain results to 'markov_chain_results.png'")


def run_predator_prey_example():
    """Run a predator-prey ecosystem simulation example."""
    print("\n--- Predator-Prey Ecosystem Simulation Example ---")
    
    # Create different types of predator-prey models
    print("Creating different predator-prey models...")
    
    # 1. Basic Lotka-Volterra model
    basic_model = create_predator_prey_model(
        model_type="basic",
        initial_prey=100,
        initial_predators=20,
        days=100
    )
    
    # 2. Model with carrying capacity
    logistic_model = create_predator_prey_model(
        model_type="logistic",
        initial_prey=100,
        initial_predators=20,
        carrying_capacity=150,
        days=100
    )
    
    # 3. Stochastic model
    stochastic_model = create_predator_prey_model(
        model_type="stochastic",
        initial_prey=100,
        initial_predators=20,
        days=100
    )
    
    # Run the simulations
    print("Running the predator-prey models...")
    basic_results = basic_model.run_simulation()
    logistic_results = logistic_model.run_simulation()
    stochastic_results = stochastic_model.run_simulation()
    
    # Calculate metrics like equilibrium points
    basic_equilibria = basic_model.get_equilibrium_points()
    logistic_equilibria = logistic_model.get_equilibrium_points()
    
    print(f"Basic model equilibrium points: {basic_equilibria}")
    print(f"Logistic model equilibrium points: {logistic_equilibria}")
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Basic model
    plt.subplot(3, 1, 1)
    plt.plot(basic_results["prey"], label='Prey')
    plt.plot(basic_results["predators"], label='Predators')
    plt.title('Basic Lotka-Volterra Model')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    # Logistic model
    plt.subplot(3, 1, 2)
    plt.plot(logistic_results["prey"], label='Prey')
    plt.plot(logistic_results["predators"], label='Predators')
    if logistic_model.carrying_capacity is not None:
        plt.axhline(y=logistic_model.carrying_capacity, color='g', linestyle='--', 
                    label=f'Carrying Capacity ({logistic_model.carrying_capacity})')
    plt.title('Lotka-Volterra Model with Carrying Capacity')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    # Stochastic model
    plt.subplot(3, 1, 3)
    plt.plot(stochastic_results["prey"], label='Prey')
    plt.plot(stochastic_results["predators"], label='Predators')
    plt.title('Stochastic Lotka-Volterra Model')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("predator_prey_results.png")
    print("Saved Predator-Prey results to 'predator_prey_results.png'")


def main():
    """Run all the advanced ecosystem simulator examples."""
    print("Running advanced ecosystem simulator examples...\n")
    
    run_network_example()
    run_markov_chain_example()
    run_predator_prey_example()


if __name__ == "__main__":
    main()