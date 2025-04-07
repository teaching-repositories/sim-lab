"""
Example of using SimLab to create an information diffusion simulation on a network.

This example demonstrates:
1. Creating different types of networks (random, scale-free, small-world)
2. Simulating information spread through the network
3. Analyzing the diffusion process 
4. Visualizing the results
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np

# Information diffusion update function for updating the network state
def information_diffusion_update(network, day):
    """Simulate how information spreads through a network.
    
    Args:
        network: The network state managed by the simulation
        day: The current simulation day
    """
    # On day 1, initialize by making some nodes "informed"
    if day == 1:
        # Seed the network with initial informed nodes
        initial_informed = 5  # First few nodes are initially informed
        for node_id in range(initial_informed):
            network.nodes[node_id].update_attribute("informed", True)
            network.nodes[node_id].update_attribute("inform_day", day)
    
    # Transmission probability - chance of spreading information to neighbors
    transmission_prob = 0.2
    
    # For each node, check if it becomes informed from neighbors
    for node_id, node in network.nodes.items():
        # Skip nodes that are already informed
        if node.attributes.get("informed", False):
            continue
        
        # Check for informed neighbors that could spread information
        for neighbor_id in node.neighbors:
            if network.nodes[neighbor_id].attributes.get("informed", False):
                # Probabilistic transmission
                if np.random.random() < transmission_prob:
                    node.update_attribute("informed", True)
                    node.update_attribute("inform_day", day)
                    break


def run_information_diffusion(network_type="scale_free", num_nodes=100, days=30):
    """Run an information diffusion simulation on a specified network type.
    
    Args:
        network_type: The type of network to create ("random", "scale_free", or "small_world")
        num_nodes: The number of nodes in the network
        days: Number of days to simulate
        
    Returns:
        Simulation results and metrics
    """
    # Create the appropriate network
    if network_type == "random":
        network = SimulatorRegistry.create(
            "NetworkSimulation",
            days=days,
            random_seed=42,
            # Use the built-in network creation function
            initial_setup={
                "type": "random",
                "num_nodes": num_nodes,
                "edge_probability": 0.05
            }
        )
    elif network_type == "scale_free":
        network = SimulatorRegistry.create(
            "NetworkSimulation",
            days=days,
            random_seed=42,
            # Use the built-in network creation function
            initial_setup={
                "type": "scale_free",
                "num_nodes": num_nodes,
                "attachment_edges": 2
            }
        )
    elif network_type == "small_world":
        network = SimulatorRegistry.create(
            "NetworkSimulation",
            days=days,
            random_seed=42,
            # Use the built-in network creation function
            initial_setup={
                "type": "small_world",
                "num_nodes": num_nodes,
                "k": 4,
                "beta": 0.1
            }
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Set the update function for information diffusion
    network.update_function = information_diffusion_update
    
    # Run the simulation
    network.run_simulation()
    
    # Calculate the diffusion curve (how many nodes are informed each day)
    informed_count = [0] * (days + 1)
    
    for day in range(1, days + 1):
        # Count nodes that are informed by this day
        count = 0
        for node in network.nodes.values():
            inform_day = node.attributes.get("inform_day", float('inf'))
            if inform_day <= day:
                count += 1
        informed_count[day] = count
    
    # Calculate network metrics
    metrics = network.calculate_metrics()
    
    return {
        "network": network,
        "informed_count": informed_count,
        "metrics": metrics
    }


# Run simulations and compare different network types
if __name__ == "__main__":
    num_nodes = 100
    days = 30
    
    print("Running information diffusion simulations on different network types...")
    
    # Run simulations on different network types
    random_results = run_information_diffusion("random", num_nodes, days)
    scale_free_results = run_information_diffusion("scale_free", num_nodes, days)
    small_world_results = run_information_diffusion("small_world", num_nodes, days)
    
    # Print some key metrics
    print("\nNetwork Metrics:")
    print("----------------")
    print(f"Random Network: Avg degree = {random_results['metrics']['avg_degree']:.2f}, "
          f"Density = {random_results['metrics']['density']:.3f}")
    print(f"Scale-Free Network: Avg degree = {scale_free_results['metrics']['avg_degree']:.2f}, "
          f"Density = {scale_free_results['metrics']['density']:.3f}")
    print(f"Small-World Network: Avg degree = {small_world_results['metrics']['avg_degree']:.2f}, "
          f"Density = {small_world_results['metrics']['density']:.3f}")
    
    # Final adoption percentages
    random_final = random_results["informed_count"][-1] / num_nodes * 100
    scale_free_final = scale_free_results["informed_count"][-1] / num_nodes * 100
    small_world_final = small_world_results["informed_count"][-1] / num_nodes * 100
    
    print("\nFinal Information Adoption:")
    print("---------------------------")
    print(f"Random Network: {random_final:.1f}% of nodes informed")
    print(f"Scale-Free Network: {scale_free_final:.1f}% of nodes informed")
    print(f"Small-World Network: {small_world_final:.1f}% of nodes informed")
    
    # Plot the diffusion curves
    plt.figure(figsize=(12, 6))
    
    # Convert to percentage of nodes
    random_pct = [count / num_nodes * 100 for count in random_results["informed_count"]]
    scale_free_pct = [count / num_nodes * 100 for count in scale_free_results["informed_count"]]
    small_world_pct = [count / num_nodes * 100 for count in small_world_results["informed_count"]]
    
    plt.plot(range(days + 1), random_pct, 'b-', marker='o', markersize=4, label='Random Network')
    plt.plot(range(days + 1), scale_free_pct, 'r-', marker='s', markersize=4, label='Scale-Free Network')
    plt.plot(range(days + 1), small_world_pct, 'g-', marker='^', markersize=4, label='Small-World Network')
    
    plt.xlabel('Day')
    plt.ylabel('Percentage of Nodes Informed')
    plt.title('Information Diffusion on Different Network Types')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation for key points
    for i, (day, pct) in enumerate(zip(range(days + 1), scale_free_pct)):
        if i > 0 and i % 5 == 0:  # Annotate every 5 days
            plt.annotate(f"{pct:.1f}%", 
                        xy=(day, pct), 
                        xytext=(day-0.3, pct+4),
                        fontsize=8)
    
    plt.tight_layout()
    plt.show()