"""
Example of using SimLab to create supply chain simulations.

This example demonstrates:
1. Building a supply chain network with factories, distributors, and retailers
2. Implementing ordering policies and demand patterns
3. Running a supply chain simulation to analyze inventory flows
4. Visualizing key performance metrics and system behavior
"""

from sim_lab.core import (
    SimulatorRegistry, SupplyChainNode, Factory, Distributor, Retailer, SupplyChainLink,
    base_stock_policy, economic_order_quantity, constant_demand, seasonal_demand, normal_demand
)
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns


def create_simple_supply_chain():
    """Create a simple linear supply chain with one node of each type."""
    # Create nodes
    factory = Factory(
        name="Factory1",
        production_capacity=100,
        production_cost=2.0,
        initial_inventory=200,
        capacity=500,
        lead_time=1
    )
    
    distributor = Distributor(
        name="Distributor1",
        shipping_cost=0.5,
        initial_inventory=150,
        capacity=300,
        lead_time=2
    )
    
    retailer = Retailer(
        name="Retailer1",
        selling_price=10.0,
        holding_cost=0.2,
        stockout_cost=2.0,
        initial_inventory=100,
        capacity=200,
        lead_time=1
    )
    
    # Create links
    factory_to_distributor = SupplyChainLink(factory, distributor)
    distributor_to_retailer = SupplyChainLink(distributor, retailer)
    
    # Create nodes dictionary
    nodes = {
        "Factory1": factory,
        "Distributor1": distributor,
        "Retailer1": retailer
    }
    
    # Create links list
    links = [factory_to_distributor, distributor_to_retailer]
    
    # Create ordering policies
    ordering_policies = {
        "Factory1": lambda node, day, sim: {},  # Factories don't order
        "Distributor1": base_stock_policy(target_level=200, review_period=1),
        "Retailer1": base_stock_policy(target_level=150, review_period=1)
    }
    
    # Create demand generator (constant demand of 20 units per day)
    demand_gen = constant_demand(rate=20)
    
    return nodes, links, ordering_policies, demand_gen


def create_multi_echelon_supply_chain():
    """Create a more complex supply chain with multiple nodes at each level."""
    # Create factories
    factory1 = Factory(
        name="Factory1",
        production_capacity=150,
        production_cost=1.8,
        initial_inventory=300,
        capacity=600,
        lead_time=2
    )
    
    factory2 = Factory(
        name="Factory2",
        production_capacity=120,
        production_cost=2.2,
        initial_inventory=250,
        capacity=500,
        lead_time=1
    )
    
    # Create distributors
    distributor1 = Distributor(
        name="Distributor1",
        shipping_cost=0.4,
        initial_inventory=200,
        capacity=400,
        lead_time=2
    )
    
    distributor2 = Distributor(
        name="Distributor2",
        shipping_cost=0.6,
        initial_inventory=180,
        capacity=350,
        lead_time=1
    )
    
    # Create retailers
    retailer1 = Retailer(
        name="Retailer1",
        selling_price=9.0,
        holding_cost=0.15,
        stockout_cost=1.8,
        initial_inventory=120,
        capacity=250,
        lead_time=1
    )
    
    retailer2 = Retailer(
        name="Retailer2",
        selling_price=10.0,
        holding_cost=0.25,
        stockout_cost=2.2,
        initial_inventory=150,
        capacity=300,
        lead_time=1
    )
    
    retailer3 = Retailer(
        name="Retailer3",
        selling_price=11.0,
        holding_cost=0.2,
        stockout_cost=2.0,
        initial_inventory=100,
        capacity=200,
        lead_time=1
    )
    
    # Create links
    links = [
        SupplyChainLink(factory1, distributor1),
        SupplyChainLink(factory1, distributor2),
        SupplyChainLink(factory2, distributor1),
        SupplyChainLink(factory2, distributor2),
        SupplyChainLink(distributor1, retailer1),
        SupplyChainLink(distributor1, retailer2),
        SupplyChainLink(distributor2, retailer2),
        SupplyChainLink(distributor2, retailer3)
    ]
    
    # Create nodes dictionary
    nodes = {
        "Factory1": factory1,
        "Factory2": factory2,
        "Distributor1": distributor1,
        "Distributor2": distributor2,
        "Retailer1": retailer1,
        "Retailer2": retailer2,
        "Retailer3": retailer3
    }
    
    # Create ordering policies
    ordering_policies = {
        "Factory1": lambda node, day, sim: {},  # Factories don't order
        "Factory2": lambda node, day, sim: {},  # Factories don't order
        "Distributor1": base_stock_policy(target_level=300, review_period=2),
        "Distributor2": economic_order_quantity(demand_rate=40, setup_cost=100, holding_cost=0.5),
        "Retailer1": base_stock_policy(target_level=180, review_period=1),
        "Retailer2": base_stock_policy(target_level=200, review_period=1),
        "Retailer3": economic_order_quantity(demand_rate=30, setup_cost=80, holding_cost=0.4)
    }
    
    # Create demand generator (different for each retailer)
    def multi_retailer_demand(day, retailer_name):
        if retailer_name == "Retailer1":
            # Constant demand
            return 25
        elif retailer_name == "Retailer2":
            # Seasonal demand
            seasonal = seasonal_demand(base_rate=30, amplitude=0.3, period=30)
            return seasonal(day, retailer_name)
        elif retailer_name == "Retailer3":
            # Normal demand
            normal = normal_demand(mean=20, std_dev=5)
            return normal(day, retailer_name)
        else:
            return 0
    
    return nodes, links, ordering_policies, multi_retailer_demand


def run_supply_chain_simulation(nodes, links, ordering_policies, demand_generator, days=100):
    """Run a supply chain simulation with the given parameters."""
    # Create the simulation
    sim = SimulatorRegistry.create(
        "SupplyChain",
        nodes=nodes,
        links=links,
        demand_generator=demand_generator,
        ordering_policies=ordering_policies,
        days=days,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    return sim, results


def visualize_inventory_levels(results, node_types=None):
    """Visualize inventory levels for all nodes or specific node types."""
    # Filter nodes by type if specified
    nodes_to_plot = {}
    for node_name, history in results.items():
        if node_name == 'overall_metrics':
            continue
            
        if node_types:
            node = None
            for n in node_types:
                if node_name.startswith(n):
                    node = node_name
                    break
            
            if node:
                nodes_to_plot[node] = history
        else:
            nodes_to_plot[node_name] = history
    
    # Determine layout based on number of nodes
    n_nodes = len(nodes_to_plot)
    if n_nodes <= 3:
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 4 * n_nodes))
        if n_nodes == 1:
            axes = [axes]  # Make it iterable
    else:
        n_rows = (n_nodes + 1) // 2  # Ceiling division
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
    
    # Plot inventory levels for each node
    for i, (node_name, history) in enumerate(nodes_to_plot.items()):
        ax = axes[i]
        days = range(len(history['inventory']))
        
        # Plot inventory
        ax.plot(days, history['inventory'], 'b-', label='Inventory', linewidth=2)
        
        # Plot backlog if available
        if 'backlog' in history:
            ax.plot(days, history['backlog'], 'r-', label='Backlog', linewidth=2)
        
        # Add labels and legend
        ax.set_title(f'Inventory Level - {node_name}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Units')
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Hide any unused subplots
    for i in range(len(nodes_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_bullwhip_effect(results):
    """Visualize the bullwhip effect across the supply chain."""
    # Extract demand and order data
    retailers = []
    distributors = []
    factories = []
    
    for node_name in results:
        if node_name == 'overall_metrics':
            continue
            
        if node_name.startswith('Retailer'):
            retailers.append(node_name)
        elif node_name.startswith('Distributor'):
            distributors.append(node_name)
        elif node_name.startswith('Factory'):
            factories.append(node_name)
    
    # Calculate total demand and orders at each level
    total_demand = np.zeros(len(results[retailers[0]]['demand']) if retailers else 0)
    total_retailer_orders = np.zeros(len(results[retailers[0]]['orders_received']) if retailers else 0)
    total_distributor_orders = np.zeros(len(results[distributors[0]]['orders_received']) if distributors else 0)
    
    for retailer in retailers:
        total_demand += np.array(results[retailer]['demand'])
        total_retailer_orders += np.array(results[retailer]['orders_received'])
    
    for distributor in distributors:
        total_distributor_orders += np.array(results[distributor]['orders_received'])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    days = range(len(total_demand))
    
    # Plot demand and orders
    plt.plot(days, total_demand, 'g-', label='Customer Demand', linewidth=2)
    plt.plot(days, total_retailer_orders, 'b-', label='Retailer Orders', linewidth=2)
    plt.plot(days, total_distributor_orders, 'r-', label='Distributor Orders', linewidth=2)
    
    # Add labels and legend
    plt.title('Bullwhip Effect: Order Variability Across Supply Chain')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate and display variability metrics
    if len(total_demand) > 0:
        demand_cv = np.std(total_demand) / np.mean(total_demand) if np.mean(total_demand) > 0 else 0
        retailer_cv = np.std(total_retailer_orders) / np.mean(total_retailer_orders) if np.mean(total_retailer_orders) > 0 else 0
        distributor_cv = np.std(total_distributor_orders) / np.mean(total_distributor_orders) if np.mean(total_distributor_orders) > 0 else 0
        
        print("\nCoefficient of Variation (CV = std/mean):")
        print(f"Customer Demand CV: {demand_cv:.4f}")
        print(f"Retailer Orders CV: {retailer_cv:.4f}")
        print(f"Distributor Orders CV: {distributor_cv:.4f}")
        print(f"Retailer/Demand CV Ratio: {retailer_cv/demand_cv:.4f}" if demand_cv > 0 else "Retailer/Demand CV Ratio: N/A")
        print(f"Distributor/Retailer CV Ratio: {distributor_cv/retailer_cv:.4f}" if retailer_cv > 0 else "Distributor/Retailer CV Ratio: N/A")


def visualize_financial_metrics(results):
    """Visualize financial metrics for the supply chain."""
    # Extract retailer data
    retailers = [node for node in results if node.startswith('Retailer')]
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # 1. Plot revenue
    ax = axes[0]
    for retailer in retailers:
        revenue = results[retailer]['revenue']
        ax.plot(range(len(revenue)), revenue, label=f'{retailer} Revenue')
    
    ax.set_title('Daily Revenue')
    ax.set_xlabel('Day')
    ax.set_ylabel('Revenue ($)')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 2. Plot costs
    ax = axes[1]
    cost_types = {
        'Production': [node for node in results if node.startswith('Factory')],
        'Shipping': [node for node in results if node.startswith('Distributor')],
        'Holding': retailers,
        'Stockout': retailers
    }
    
    for cost_type, nodes in cost_types.items():
        if cost_type == 'Production':
            costs = np.zeros(len(results[nodes[0]]['production_cost']) if nodes else 0)
            for node in nodes:
                costs += np.array(results[node]['production_cost'])
        elif cost_type == 'Shipping':
            costs = np.zeros(len(results[nodes[0]]['shipping_cost']) if nodes else 0)
            for node in nodes:
                costs += np.array(results[node]['shipping_cost'])
        elif cost_type == 'Holding':
            costs = np.zeros(len(results[nodes[0]]['holding_cost']) if nodes else 0)
            for node in nodes:
                costs += np.array(results[node]['holding_cost'])
        elif cost_type == 'Stockout':
            costs = np.zeros(len(results[nodes[0]]['stockout_cost']) if nodes else 0)
            for node in nodes:
                costs += np.array(results[node]['stockout_cost'])
        
        ax.plot(range(len(costs)), costs, label=f'{cost_type} Cost')
    
    ax.set_title('Daily Costs by Type')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cost ($)')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 3. Plot service level (fulfilled/demanded)
    ax = axes[2]
    combined_demand = np.zeros(len(results[retailers[0]]['demand']) if retailers else 0)
    combined_sales = np.zeros(len(results[retailers[0]]['sales']) if retailers else 0)
    
    for retailer in retailers:
        combined_demand += np.array(results[retailer]['demand'])
        combined_sales += np.array(results[retailer]['sales'])
    
    service_level = combined_sales / combined_demand
    service_level = np.nan_to_num(service_level, nan=1.0)  # Replace NaN with 1.0
    
    ax.plot(range(len(service_level)), service_level, 'g-', linewidth=2)
    ax.axhline(y=np.mean(service_level), color='r', linestyle='--', 
               label=f'Average: {np.mean(service_level):.2%}')
    
    ax.set_title('Daily Service Level')
    ax.set_xlabel('Day')
    ax.set_ylabel('Service Level (%)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 4. Plot overall metrics
    ax = axes[3]
    metrics = results['overall_metrics']
    
    # Create a horizontal bar chart
    metric_names = ['Total Revenue', 'Total Costs', 'Total Profit', 'Service Level']
    metric_values = [
        metrics['total_revenue'][0],
        metrics['total_costs'][0],
        metrics['total_profit'][0],
        metrics['service_level'][0] * 100  # Convert to percentage
    ]
    
    colors = ['green', 'red', 'blue', 'purple']
    y_pos = range(len(metric_names))
    
    ax.barh(y_pos, metric_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    
    # Add values to the bars
    for i, v in enumerate(metric_values):
        if i < 3:  # Financial metrics
            ax.text(v + 0.1, i, f'${v:.2f}', va='center')
        else:  # Service level
            ax.text(v + 0.1, i, f'{v:.2f}%', va='center')
    
    ax.set_title('Overall Supply Chain Performance')
    ax.set_xlabel('Value')
    
    plt.tight_layout()
    plt.show()


def visualize_network_diagram(nodes, links):
    """Visualize the supply chain network structure."""
    plt.figure(figsize=(12, 8))
    
    # Define node positions (simple layout)
    node_positions = {}
    
    # Identify node types and counts
    factories = [name for name in nodes if name.startswith("Factory")]
    distributors = [name for name in nodes if name.startswith("Distributor")]
    retailers = [name for name in nodes if name.startswith("Retailer")]
    
    # Position factories on left
    for i, factory in enumerate(factories):
        y_pos = (i - (len(factories) - 1) / 2) * 2
        node_positions[factory] = (0, y_pos)
    
    # Position distributors in middle
    for i, distributor in enumerate(distributors):
        y_pos = (i - (len(distributors) - 1) / 2) * 2
        node_positions[distributor] = (5, y_pos)
    
    # Position retailers on right
    for i, retailer in enumerate(retailers):
        y_pos = (i - (len(retailers) - 1) / 2) * 2
        node_positions[retailer] = (10, y_pos)
    
    # Define node colors and sizes
    node_colors = {
        'Factory': 'green',
        'Distributor': 'blue',
        'Retailer': 'red'
    }
    
    node_sizes = {
        'Factory': 1500,
        'Distributor': 1200,
        'Retailer': 1000
    }
    
    # Draw nodes
    for name, node in nodes.items():
        x, y = node_positions[name]
        node_type = name.split('1')[0]  # Extract node type from name
        
        plt.scatter(x, y, s=node_sizes[node_type], color=node_colors[node_type], 
                   edgecolor='black', zorder=10)
        plt.text(x, y, name, ha='center', va='center', fontweight='bold')
    
    # Draw links
    for link in links:
        source_name = link.source.name
        dest_name = link.destination.name
        
        source_pos = node_positions[source_name]
        dest_pos = node_positions[dest_name]
        
        plt.plot([source_pos[0], dest_pos[0]], [source_pos[1], dest_pos[1]], 
                'k-', linewidth=1.5, alpha=0.6)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Factory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Distributor'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Retailer')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    plt.title('Supply Chain Network Structure')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_ordering_policies():
    """Compare different ordering policies in the same supply chain."""
    # Create a simple supply chain as a base
    nodes_base, links_base, _, demand_gen = create_simple_supply_chain()
    
    # Define different ordering policies to compare
    policies = {
        "Base Stock": {
            "Factory1": lambda node, day, sim: {},  # Factories don't order
            "Distributor1": base_stock_policy(target_level=200, review_period=1),
            "Retailer1": base_stock_policy(target_level=150, review_period=1)
        },
        "EOQ": {
            "Factory1": lambda node, day, sim: {},  # Factories don't order
            "Distributor1": economic_order_quantity(demand_rate=25, setup_cost=100, holding_cost=0.5),
            "Retailer1": economic_order_quantity(demand_rate=20, setup_cost=80, holding_cost=0.3)
        },
        "Hybrid": {
            "Factory1": lambda node, day, sim: {},  # Factories don't order
            "Distributor1": base_stock_policy(target_level=200, review_period=2),
            "Retailer1": economic_order_quantity(demand_rate=20, setup_cost=80, holding_cost=0.3)
        }
    }
    
    # Run simulations with different policies
    sim_results = {}
    for policy_name, ordering_policy in policies.items():
        # We need to create a fresh copy of nodes and links for each run
        nodes_copy = {
            "Factory1": Factory(
                name="Factory1",
                production_capacity=100,
                production_cost=2.0,
                initial_inventory=200,
                capacity=500,
                lead_time=1
            ),
            "Distributor1": Distributor(
                name="Distributor1",
                shipping_cost=0.5,
                initial_inventory=150,
                capacity=300,
                lead_time=2
            ),
            "Retailer1": Retailer(
                name="Retailer1",
                selling_price=10.0,
                holding_cost=0.2,
                stockout_cost=2.0,
                initial_inventory=100,
                capacity=200,
                lead_time=1
            )
        }
        
        links_copy = [
            SupplyChainLink(nodes_copy["Factory1"], nodes_copy["Distributor1"]),
            SupplyChainLink(nodes_copy["Distributor1"], nodes_copy["Retailer1"])
        ]
        
        sim, results = run_supply_chain_simulation(
            nodes_copy, links_copy, ordering_policy, demand_gen, days=100
        )
        
        sim_results[policy_name] = results
    
    # Compare results
    # 1. Service level comparison
    plt.figure(figsize=(12, 6))
    
    for policy_name, results in sim_results.items():
        # Calculate service level over time
        retailer = results['Retailer1']
        demand = np.array(retailer['demand'])
        sales = np.array(retailer['sales'])
        
        service_level = sales / demand
        service_level = np.nan_to_num(service_level, nan=1.0)  # Replace NaN with 1.0
        
        # Calculate rolling average for smoother visualization
        window_size = 5
        rolling_avg = np.convolve(service_level, np.ones(window_size)/window_size, mode='valid')
        days = range(len(rolling_avg))
        
        plt.plot(days, rolling_avg, label=f'{policy_name}', linewidth=2)
    
    plt.title('Service Level Comparison by Ordering Policy')
    plt.xlabel('Day')
    plt.ylabel('Service Level')
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Inventory level comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Distributor inventory
    ax = axes[0]
    for policy_name, results in sim_results.items():
        inventory = results['Distributor1']['inventory']
        ax.plot(range(len(inventory)), inventory, label=f'{policy_name}', linewidth=2)
    
    ax.set_title('Distributor Inventory by Ordering Policy')
    ax.set_xlabel('Day')
    ax.set_ylabel('Inventory Level')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Retailer inventory
    ax = axes[1]
    for policy_name, results in sim_results.items():
        inventory = results['Retailer1']['inventory']
        ax.plot(range(len(inventory)), inventory, label=f'{policy_name}', linewidth=2)
    
    ax.set_title('Retailer Inventory by Ordering Policy')
    ax.set_xlabel('Day')
    ax.set_ylabel('Inventory Level')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Overall performance metrics comparison
    metrics = {
        'Total Profit': [],
        'Service Level': [],
        'Avg Inventory': []
    }
    
    policy_names = list(sim_results.keys())
    
    for policy_name, results in sim_results.items():
        # Profit
        metrics['Total Profit'].append(results['overall_metrics']['total_profit'][0])
        
        # Service level
        metrics['Service Level'].append(results['overall_metrics']['service_level'][0] * 100)  # as percentage
        
        # Average inventory across all nodes
        avg_inventories = []
        for node_name, node_results in results.items():
            if node_name != 'overall_metrics':
                avg_inventories.append(np.mean(node_results['inventory']))
        
        metrics['Avg Inventory'].append(np.mean(avg_inventories))
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Profit comparison
    ax = axes[0]
    ax.bar(policy_names, metrics['Total Profit'], color='green')
    ax.set_title('Total Profit')
    ax.set_ylabel('Profit ($)')
    for i, v in enumerate(metrics['Total Profit']):
        ax.text(i, v + 10, f'${v:.2f}', ha='center')
    
    # Service level comparison
    ax = axes[1]
    ax.bar(policy_names, metrics['Service Level'], color='blue')
    ax.set_title('Service Level')
    ax.set_ylabel('Service Level (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(metrics['Service Level']):
        ax.text(i, v + 2, f'{v:.2f}%', ha='center')
    
    # Average inventory comparison
    ax = axes[2]
    ax.bar(policy_names, metrics['Avg Inventory'], color='orange')
    ax.set_title('Average Inventory')
    ax.set_ylabel('Units')
    for i, v in enumerate(metrics['Avg Inventory']):
        ax.text(i, v + 2, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return sim_results


# Run examples
if __name__ == "__main__":
    print("1. Running Simple Supply Chain Simulation...")
    nodes, links, ordering_policies, demand_gen = create_simple_supply_chain()
    
    # Visualize the network
    visualize_network_diagram(nodes, links)
    
    # Run the simulation
    sim, results = run_supply_chain_simulation(nodes, links, ordering_policies, demand_gen, days=100)
    
    # Visualize inventory levels
    visualize_inventory_levels(results)
    
    # Visualize financial metrics
    visualize_financial_metrics(results)
    
    print("\n2. Running Multi-Echelon Supply Chain Simulation...")
    nodes, links, ordering_policies, demand_gen = create_multi_echelon_supply_chain()
    
    # Visualize the network
    visualize_network_diagram(nodes, links)
    
    # Run the simulation
    sim, results = run_supply_chain_simulation(nodes, links, ordering_policies, demand_gen, days=100)
    
    # Visualize inventory levels for retailers
    visualize_inventory_levels(results, node_types=["Retailer"])
    
    # Visualize the bullwhip effect
    visualize_bullwhip_effect(results)
    
    # Visualize financial metrics
    visualize_financial_metrics(results)
    
    print("\n3. Comparing Different Ordering Policies...")
    policy_results = compare_ordering_policies()
    
    # Print overall outcomes
    print("\nComparison Summary:")
    
    for policy_name, results in policy_results.items():
        profit = results['overall_metrics']['total_profit'][0]
        service_level = results['overall_metrics']['service_level'][0] * 100  # as percentage
        
        print(f"{policy_name} Policy:")
        print(f"  - Total Profit: ${profit:.2f}")
        print(f"  - Service Level: {service_level:.2f}%")