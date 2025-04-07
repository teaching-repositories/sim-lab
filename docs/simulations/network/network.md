# Network Simulation

The Network Simulation module in SimLab allows you to model processes that occur on complex networks or graphs. This includes phenomena like information diffusion, disease spread, opinion formation, and many other network-based processes.

## Overview

The NetworkSimulation class provides a flexible framework for simulating dynamic processes on networks with different topologies. Key features include:

- Directed and undirected networks
- Weighted edges
- Dynamic network evolution
- Support for node and edge attributes
- Built-in network metrics
- Pre-defined network generation models (Random, Scale-Free, Small-World)

## Basic Usage

Here's a simple example of creating and using a network simulation:

```python
from sim_lab.core import NetworkSimulation, create_random_network

# Create a random network with 50 nodes and 10% edge probability
network = create_random_network(
    num_nodes=50,
    edge_probability=0.1,
    directed=False
)

# Define a custom update function
def update_function(network, day):
    # Example: Mark random nodes as "active"
    for node_id, node in network.nodes.items():
        if np.random.random() < 0.1:  # 10% chance
            node.update_attribute("active", True)

# Set the update function
network.update_function = update_function

# Run the simulation for 100 days
metrics = network.run_simulation()

# Analyze the results
num_active = sum(1 for node in network.nodes.values() 
                 if node.attributes.get("active", False))
print(f"Number of active nodes: {num_active}")
```

## Pre-defined Network Models

SimLab provides built-in functions to create common network types:

### Random Network (Erdős–Rényi Model)

```python
from sim_lab.core import create_random_network

random_net = create_random_network(
    num_nodes=50,
    edge_probability=0.1,
    directed=False
)
```

### Scale-Free Network (Barabási–Albert Model)

```python
from sim_lab.core import create_scale_free_network

scale_free_net = create_scale_free_network(
    num_nodes=50,
    m=2,  # Each new node attaches to m existing nodes
    directed=False
)
```

### Small-World Network (Watts–Strogatz Model)

```python
from sim_lab.core import create_small_world_network

small_world_net = create_small_world_network(
    num_nodes=50,
    k=4,  # Each node connects to k nearest neighbors
    beta=0.1,  # Rewiring probability
    directed=False
)
```

## Network Analysis

The NetworkSimulation class provides several methods for analyzing the network structure:

```python
# Get the adjacency matrix representation
adj_matrix = network.get_adjacency_matrix()

# Get the degree distribution
degree_dist = network.get_degree_distribution()

# Calculate network metrics
metrics = network.calculate_metrics()
print(f"Average degree: {metrics['avg_degree']}")
print(f"Network density: {metrics['density']}")
```

## Example: Information Diffusion

This example shows how to simulate information diffusion through a network:

```python
from sim_lab.core import create_scale_free_network
import numpy as np
import matplotlib.pyplot as plt

# Create a scale-free network
network = create_scale_free_network(num_nodes=100, m=2)

# Information diffusion update function
def diffusion_update(network, day):
    # On day 1, seed the network with initial "informed" nodes
    if day == 1:
        for node_id in range(5):  # First 5 nodes are initially informed
            network.nodes[node_id].update_attribute("informed", True)
            network.nodes[node_id].update_attribute("inform_day", day)
    
    # For all other days, spread the information
    transmission_prob = 0.2  # Probability of spreading information
    
    for node_id, node in network.nodes.items():
        # Skip nodes that are already informed
        if node.attributes.get("informed", False):
            continue
        
        # Check if any neighbors are informed
        for neighbor_id in node.neighbors:
            if network.nodes[neighbor_id].attributes.get("informed", False):
                if np.random.random() < transmission_prob:
                    node.update_attribute("informed", True)
                    node.update_attribute("inform_day", day)
                    break

# Set the update function
network.update_function = diffusion_update

# Run the simulation
network.run_simulation()

# Calculate the spread
days = network.days
informed_count = [0] * days
for day in range(days):
    count = sum(1 for node in network.nodes.values() 
                if node.attributes.get("informed", False) and 
                   node.attributes.get("inform_day", float('inf')) <= day)
    informed_count[day] = count

# Plot the diffusion curve
plt.figure(figsize=(10, 6))
plt.plot(informed_count)
plt.title('Information Diffusion on Scale-Free Network')
plt.xlabel('Day')
plt.ylabel('Number of Informed Nodes')
plt.grid(True)
plt.show()
```

## API Reference

### NetworkSimulation Class

```python
NetworkSimulation(
    initial_nodes: Optional[Dict[Any, Dict[str, Any]]] = None,
    initial_edges: Optional[List[Tuple[Any, Any, Dict[str, Any]]]] = None,
    update_function: Optional[Callable] = None,
    directed: bool = False,
    days: int = 100,
    save_history: bool = False,
    random_seed: Optional[int] = None
)
```

#### Parameters

- **initial_nodes**: Dictionary mapping node IDs to attribute dictionaries
- **initial_edges**: List of (source, target, attributes) tuples
- **update_function**: Function that updates the network at each time step
- **directed**: Whether the network is directed
- **days**: Number of steps to simulate
- **save_history**: Whether to save node and edge history
- **random_seed**: Seed for random number generation

#### Methods

- **add_node(node_id, attributes)**: Add a node to the network
- **remove_node(node_id)**: Remove a node from the network
- **add_edge(source, target, directed, weight, attributes)**: Add an edge to the network
- **remove_edge(source, target)**: Remove an edge from the network
- **get_adjacency_matrix()**: Get the adjacency matrix of the network
- **get_degree_distribution()**: Get the degree distribution of the network
- **calculate_metrics()**: Calculate metrics for the current network state
- **run_simulation()**: Run the network simulation
- **get_node_attribute_history(node_id, attribute)**: Get history of a node attribute
- **get_edge_attribute_history(source, target, attribute)**: Get history of an edge attribute

### Node Class

```python
Node(node_id: Any, attributes: Optional[Dict[str, Any]] = None)
```

#### Methods

- **add_neighbor(neighbor_id)**: Add a neighbor to this node
- **remove_neighbor(neighbor_id)**: Remove a neighbor from this node
- **update_attribute(key, value)**: Update a node attribute
- **get_attribute_history(attribute)**: Get history of an attribute

### Edge Class

```python
Edge(source: Any, target: Any, directed: bool = False, weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None)
```

#### Methods

- **update_weight(weight)**: Update the edge weight
- **update_attribute(key, value)**: Update an edge attribute
- **get_attribute_history(attribute)**: Get history of an attribute

## Advanced Topics

### Custom Network Topology

You can create a custom network topology by manually specifying nodes and edges:

```python
# Create a star network topology
nodes = {i: {} for i in range(6)}  # Center node + 5 peripheral nodes
edges = []

# Connect center node (0) to all others
for i in range(1, 6):
    edges.append((0, i, {'weight': 1.0}))

network = NetworkSimulation(
    initial_nodes=nodes,
    initial_edges=edges,
    directed=False
)
```

### Dynamic Network Evolution

You can implement dynamic network evolution by modifying the network structure in the update function:

```python
def dynamic_network_update(network, day):
    # Add a new node with some probability
    if np.random.random() < 0.1:
        new_id = max(network.nodes.keys()) + 1
        network.add_node(new_id, {'creation_day': day})
        
        # Connect to 2 random existing nodes
        existing_nodes = list(network.nodes.keys())
        if len(existing_nodes) > 0:
            for _ in range(min(2, len(existing_nodes))):
                target = np.random.choice(existing_nodes)
                network.add_edge(new_id, target)
    
    # Remove a node with some probability
    if len(network.nodes) > 10 and np.random.random() < 0.05:
        node_to_remove = np.random.choice(list(network.nodes.keys()))
        network.remove_node(node_to_remove)
```

### Weighted Networks

You can specify edge weights when creating edges:

```python
# Create a weighted network
nodes = {i: {} for i in range(5)}
edges = []

# Add edges with different weights
edges.append((0, 1, {'weight': 0.5}))
edges.append((0, 2, {'weight': 1.0}))
edges.append((1, 2, {'weight': 2.0}))
edges.append((1, 3, {'weight': 0.7}))
edges.append((2, 4, {'weight': 1.5}))

weighted_network = NetworkSimulation(
    initial_nodes=nodes,
    initial_edges=edges
)
```

## Further Reading

For more information about network science and related topics, see:

- [Network Science by Albert-László Barabási](http://networksciencebook.com/)
- [NetworkX documentation](https://networkx.org/) (a popular Python library for network analysis)
- [Social Network Analysis: Methods and Applications](https://www.cambridge.org/core/books/social-network-analysis/20A8013A6B93E600C1F3ACB5FC6334A4) by Wasserman and Faust