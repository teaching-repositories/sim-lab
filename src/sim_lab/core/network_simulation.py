"""Network/Graph Simulation implementation."""

import numpy as np
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


class Node:
    """Represents a node in a network graph.
    
    Attributes:
        node_id (Any): Unique identifier for the node.
        attributes (Dict[str, Any]): Node attributes.
        neighbors (Set[Any]): Set of neighbor node IDs.
    """
    
    def __init__(self, node_id: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a new node.
        
        Args:
            node_id: Unique identifier for the node.
            attributes: Dictionary of node attributes.
        """
        self.node_id = node_id
        self.attributes = attributes or {}
        self.neighbors = set()
        self.history = []  # Track attribute history
    
    def add_neighbor(self, neighbor_id: Any) -> None:
        """Add a neighbor to this node.
        
        Args:
            neighbor_id: The ID of the neighbor node.
        """
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: Any) -> None:
        """Remove a neighbor from this node.
        
        Args:
            neighbor_id: The ID of the neighbor node.
        """
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
    
    def update_attribute(self, key: str, value: Any) -> None:
        """Update a node attribute.
        
        Args:
            key: The attribute key.
            value: The new attribute value.
        """
        self.attributes[key] = value
    
    def save_history(self) -> None:
        """Save the current state to history."""
        self.history.append(self.attributes.copy())
    
    def get_attribute_history(self, attribute: str) -> List[Any]:
        """Get the history of a specific attribute.
        
        Args:
            attribute: The name of the attribute.
            
        Returns:
            List of values for the attribute over time.
        """
        return [state.get(attribute) for state in self.history if attribute in state]


class Edge:
    """Represents an edge in a network graph.
    
    Attributes:
        source (Any): Source node ID.
        target (Any): Target node ID.
        directed (bool): Whether the edge is directed.
        weight (float): Edge weight.
        attributes (Dict[str, Any]): Edge attributes.
    """
    
    def __init__(
        self, 
        source: Any, 
        target: Any, 
        directed: bool = False,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a new edge.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            directed: Whether the edge is directed.
            weight: Edge weight.
            attributes: Dictionary of edge attributes.
        """
        self.source = source
        self.target = target
        self.directed = directed
        self.weight = weight
        self.attributes = attributes or {}
        self.history = []  # Track attribute history
    
    def update_weight(self, weight: float) -> None:
        """Update the edge weight.
        
        Args:
            weight: The new weight.
        """
        self.weight = weight
        self.attributes['weight'] = weight
    
    def update_attribute(self, key: str, value: Any) -> None:
        """Update an edge attribute.
        
        Args:
            key: The attribute key.
            value: The new attribute value.
        """
        self.attributes[key] = value
    
    def save_history(self) -> None:
        """Save the current state to history."""
        state = self.attributes.copy()
        state['weight'] = self.weight
        self.history.append(state)
    
    def get_attribute_history(self, attribute: str) -> List[Any]:
        """Get the history of a specific attribute.
        
        Args:
            attribute: The name of the attribute.
            
        Returns:
            List of values for the attribute over time.
        """
        if attribute == 'weight':
            return [state.get('weight', self.weight) for state in self.history]
        return [state.get(attribute) for state in self.history if attribute in state]


@SimulatorRegistry.register("Network")
class NetworkSimulation(BaseSimulation):
    """A simulation class for network/graph dynamics.
    
    This simulation models the evolution of a network over time, allowing for changes 
    in node and edge attributes, as well as network structure.
    
    Attributes:
        nodes (Dict[Any, Node]): Dictionary of nodes in the network.
        edges (List[Edge]): List of edges in the network.
        days (int): Number of steps to simulate.
        update_function (Callable): Function to update the network at each time step.
        save_history (bool): Whether to save node and edge history.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self,
        initial_nodes: Optional[Dict[Any, Dict[str, Any]]] = None,
        initial_edges: Optional[List[Tuple[Any, Any, Dict[str, Any]]]] = None,
        update_function: Optional[Callable] = None,
        directed: bool = False,
        days: int = 100,
        save_history: bool = False,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the network simulation.
        
        Args:
            initial_nodes: Dictionary mapping node IDs to attribute dictionaries.
            initial_edges: List of (source, target, attributes) tuples.
            update_function: Function that updates the network at each time step.
            directed: Whether the network is directed.
            days: Number of steps to simulate.
            save_history: Whether to save node and edge history.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        self.directed = directed
        self.save_history = save_history
        self.update_function = update_function or (lambda network, day: None)
        
        # Initialize nodes
        self.nodes = {}
        if initial_nodes:
            for node_id, attributes in initial_nodes.items():
                self.add_node(node_id, attributes)
        
        # Initialize edges
        self.edges = []
        if initial_edges:
            for source, target, attributes in initial_edges:
                weight = attributes.pop('weight', 1.0) if attributes else 1.0
                self.add_edge(source, target, directed, weight, attributes)
                
        # Initialize metrics tracking
        self.metrics = {}
    
    def add_node(self, node_id: Any, attributes: Optional[Dict[str, Any]] = None) -> Node:
        """Add a node to the network.
        
        Args:
            node_id: Unique identifier for the node.
            attributes: Dictionary of node attributes.
            
        Returns:
            The created node.
            
        Raises:
            ValueError: If a node with the given ID already exists.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with ID {node_id} already exists")
        
        node = Node(node_id, attributes)
        self.nodes[node_id] = node
        return node
    
    def remove_node(self, node_id: Any) -> None:
        """Remove a node from the network.
        
        Args:
            node_id: The ID of the node to remove.
            
        Raises:
            ValueError: If the node doesn't exist.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} doesn't exist")
        
        # Remove edges connected to this node
        self.edges = [edge for edge in self.edges if edge.source != node_id and edge.target != node_id]
        
        # Remove node from neighbor lists
        for node in self.nodes.values():
            if node_id in node.neighbors:
                node.neighbors.remove(node_id)
        
        # Remove the node
        del self.nodes[node_id]
    
    def add_edge(
        self,
        source: Any,
        target: Any,
        directed: Optional[bool] = None,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Edge:
        """Add an edge to the network.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            directed: Whether the edge is directed (defaults to network's directed attribute).
            weight: Edge weight.
            attributes: Dictionary of edge attributes.
            
        Returns:
            The created edge.
            
        Raises:
            ValueError: If the source or target node doesn't exist.
        """
        if source not in self.nodes:
            raise ValueError(f"Source node with ID {source} doesn't exist")
        if target not in self.nodes:
            raise ValueError(f"Target node with ID {target} doesn't exist")
        
        # Use network's directed attribute if not specified
        if directed is None:
            directed = self.directed
        
        # Create the edge
        edge = Edge(source, target, directed, weight, attributes)
        self.edges.append(edge)
        
        # Update node neighbor lists
        self.nodes[source].add_neighbor(target)
        if not directed:
            self.nodes[target].add_neighbor(source)
        
        return edge
    
    def remove_edge(self, source: Any, target: Any) -> None:
        """Remove an edge from the network.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            
        Raises:
            ValueError: If the edge doesn't exist.
        """
        # Find the edge
        for i, edge in enumerate(self.edges):
            if edge.source == source and edge.target == target:
                # Remove from neighbor lists
                self.nodes[source].remove_neighbor(target)
                if not edge.directed:
                    self.nodes[target].remove_neighbor(source)
                
                # Remove the edge
                self.edges.pop(i)
                return
                
        # Check for undirected edge in reverse direction
        if not self.directed:
            for i, edge in enumerate(self.edges):
                if edge.source == target and edge.target == source:
                    # Remove from neighbor lists
                    self.nodes[target].remove_neighbor(source)
                    self.nodes[source].remove_neighbor(target)
                    
                    # Remove the edge
                    self.edges.pop(i)
                    return
        
        raise ValueError(f"Edge from {source} to {target} doesn't exist")
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix of the network.
        
        Returns:
            A NumPy array representing the adjacency matrix, with weights if applicable.
        """
        # Create a mapping from node IDs to indices
        node_ids = list(self.nodes.keys())
        node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Initialize the adjacency matrix
        n = len(node_ids)
        adj_matrix = np.zeros((n, n))
        
        # Fill in the adjacency matrix
        for edge in self.edges:
            i = node_to_index[edge.source]
            j = node_to_index[edge.target]
            adj_matrix[i, j] = edge.weight
            if not edge.directed:
                adj_matrix[j, i] = edge.weight
        
        return adj_matrix
    
    def get_degree_distribution(self) -> Dict[int, int]:
        """Get the degree distribution of the network.
        
        Returns:
            A dictionary mapping degrees to the number of nodes with that degree.
        """
        degrees = [len(node.neighbors) for node in self.nodes.values()]
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        return degree_counts
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current network state.
        
        Returns:
            A dictionary of network metrics.
        """
        # Number of nodes and edges
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        
        # Average degree
        total_degree = sum(len(node.neighbors) for node in self.nodes.values())
        avg_degree = total_degree / num_nodes if num_nodes > 0 else 0
        
        # Density (ratio of actual to possible edges)
        possible_edges = num_nodes * (num_nodes - 1)
        if self.directed:
            density = num_edges / possible_edges if possible_edges > 0 else 0
        else:
            density = 2 * num_edges / possible_edges if possible_edges > 0 else 0
        
        # Result dictionary
        metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density': density
        }
        
        return metrics
    
    def run_simulation(self) -> List[Dict[str, Any]]:
        """Run the network simulation.
        
        In each step, the network is updated according to the update function.
        
        Returns:
            A list of dictionaries containing network metrics for each time step.
        """
        self.reset()
        
        # Initialize history if tracking
        if self.save_history:
            for node in self.nodes.values():
                node.history = [node.attributes.copy()]
            
            for edge in self.edges:
                state = edge.attributes.copy()
                state['weight'] = edge.weight
                edge.history = [state]
        
        # Calculate initial metrics
        self.metrics = [self.calculate_metrics()]
        
        # Run for specified number of days
        for day in range(1, self.days):
            # Update the network
            self.update_function(self, day)
            
            # Save history if tracking
            if self.save_history:
                for node in self.nodes.values():
                    node.save_history()
                
                for edge in self.edges:
                    edge.save_history()
            
            # Calculate metrics
            self.metrics.append(self.calculate_metrics())
        
        return self.metrics
    
    def get_node_attribute_history(self, node_id: Any, attribute: str) -> List[Any]:
        """Get the history of a specific node attribute.
        
        Args:
            node_id: The ID of the node.
            attribute: The name of the attribute.
            
        Returns:
            List of values for the attribute over time.
            
        Raises:
            ValueError: If the node doesn't exist or history wasn't saved.
        """
        if not self.save_history:
            raise ValueError("Node history wasn't saved. Set save_history=True when creating the simulation.")
        
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID {node_id} doesn't exist")
        
        return self.nodes[node_id].get_attribute_history(attribute)
    
    def get_edge_attribute_history(self, source: Any, target: Any, attribute: str) -> List[Any]:
        """Get the history of a specific edge attribute.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            attribute: The name of the attribute.
            
        Returns:
            List of values for the attribute over time.
            
        Raises:
            ValueError: If the edge doesn't exist or history wasn't saved.
        """
        if not self.save_history:
            raise ValueError("Edge history wasn't saved. Set save_history=True when creating the simulation.")
        
        # Find the edge
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge.get_attribute_history(attribute)
        
        # Check for undirected edge in reverse direction
        if not self.directed:
            for edge in self.edges:
                if edge.source == target and edge.target == source:
                    return edge.get_attribute_history(attribute)
        
        raise ValueError(f"Edge from {source} to {target} doesn't exist")
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        
        # Clear metrics
        self.metrics = []
        
        # Clear node and edge history
        if self.save_history:
            for node in self.nodes.values():
                node.history = [node.attributes.copy()]
            
            for edge in self.edges:
                state = edge.attributes.copy()
                state['weight'] = edge.weight
                edge.history = [state]
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata.
        """
        # Get base parameters from parent class
        params = super().get_parameters_info()
        
        # Add class-specific parameters
        params.update({
            'initial_nodes': {
                'type': 'Dict[Any, Dict[str, Any]]',
                'description': 'Dictionary mapping node IDs to attribute dictionaries',
                'required': False,
                'default': '{}'
            },
            'initial_edges': {
                'type': 'List[Tuple[Any, Any, Dict[str, Any]]]',
                'description': 'List of (source, target, attributes) tuples',
                'required': False,
                'default': '[]'
            },
            'update_function': {
                'type': 'Callable',
                'description': 'Function that updates the network at each time step',
                'required': False,
                'default': 'None'
            },
            'directed': {
                'type': 'bool',
                'description': 'Whether the network is directed',
                'required': False,
                'default': 'False'
            },
            'save_history': {
                'type': 'bool',
                'description': 'Whether to save node and edge history',
                'required': False,
                'default': 'False'
            }
        })
        
        return params


# Common network models
def create_random_network(num_nodes: int, edge_probability: float, directed: bool = False) -> NetworkSimulation:
    """Create an Erdős–Rényi random network.
    
    Args:
        num_nodes: Number of nodes in the network.
        edge_probability: Probability of creating an edge between any two nodes.
        directed: Whether the network is directed.
        
    Returns:
        A NetworkSimulation with a random network structure.
    """
    # Create initial nodes
    initial_nodes = {i: {} for i in range(num_nodes)}
    
    # Create initial edges
    initial_edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes) if not directed else range(num_nodes):
            if i != j and random.random() < edge_probability:
                initial_edges.append((i, j, {'weight': 1.0}))
    
    # Create the simulation
    return NetworkSimulation(
        initial_nodes=initial_nodes,
        initial_edges=initial_edges,
        directed=directed
    )


def create_scale_free_network(num_nodes: int, m: int = 2, directed: bool = False) -> NetworkSimulation:
    """Create a Barabási–Albert scale-free network.
    
    This model generates a network with a power-law degree distribution.
    
    Args:
        num_nodes: Number of nodes in the network.
        m: Number of edges to add for each new node.
        directed: Whether the network is directed.
        
    Returns:
        A NetworkSimulation with a scale-free network structure.
    """
    if num_nodes <= m:
        raise ValueError(f"Number of nodes ({num_nodes}) must be greater than m ({m})")
    
    # Create the simulation with initial nodes
    initial_nodes = {i: {} for i in range(num_nodes)}
    
    # Create initial edges for a fully connected graph of m nodes
    initial_edges = []
    for i in range(m):
        for j in range(i + 1, m):
            initial_edges.append((i, j, {'weight': 1.0}))
    
    # Add remaining nodes with preferential attachment
    degrees = [0] * num_nodes
    for i in range(m):
        for j in range(m):
            if i != j:
                degrees[i] += 1
    
    for i in range(m, num_nodes):
        # Select m nodes to connect to, with probability proportional to degree
        targets = []
        while len(targets) < m:
            target = _select_target(degrees[:i], targets)
            if target not in targets:
                targets.append(target)
        
        # Add edges to the selected targets
        for target in targets:
            initial_edges.append((i, target, {'weight': 1.0}))
            degrees[i] += 1
            degrees[target] += 1
    
    # Create the simulation
    return NetworkSimulation(
        initial_nodes=initial_nodes,
        initial_edges=initial_edges,
        directed=directed
    )


def _select_target(degrees: List[int], exclude: List[int]) -> int:
    """Select a target node for preferential attachment.
    
    Args:
        degrees: List of node degrees.
        exclude: List of nodes to exclude.
        
    Returns:
        The index of the selected node.
    """
    total_degree = sum(degrees[i] for i in range(len(degrees)) if i not in exclude)
    if total_degree == 0:
        # If all nodes have degree 0, select uniformly
        while True:
            target = random.randint(0, len(degrees) - 1)
            if target not in exclude:
                return target
    
    # Select with probability proportional to degree
    r = random.random() * total_degree
    cumulative = 0
    for i in range(len(degrees)):
        if i not in exclude:
            cumulative += degrees[i]
            if cumulative >= r:
                return i
    
    # Fallback (should not reach here)
    for i in range(len(degrees)):
        if i not in exclude:
            return i
    
    raise ValueError("No valid target found")


def create_small_world_network(num_nodes: int, k: int = 4, beta: float = 0.1, directed: bool = False) -> NetworkSimulation:
    """Create a Watts–Strogatz small-world network.
    
    This model generates a network with high clustering and small average path length.
    
    Args:
        num_nodes: Number of nodes in the network.
        k: Each node is connected to k nearest neighbors in ring topology (must be even).
        beta: Probability of rewiring each edge.
        directed: Whether the network is directed.
        
    Returns:
        A NetworkSimulation with a small-world network structure.
    """
    if k % 2 != 0:
        raise ValueError(f"k ({k}) must be even")
    if k >= num_nodes:
        raise ValueError(f"k ({k}) must be less than num_nodes ({num_nodes})")
    
    # Create initial nodes
    initial_nodes = {i: {} for i in range(num_nodes)}
    
    # Create initial edges: connect each node to its k nearest neighbors
    initial_edges = []
    for i in range(num_nodes):
        for j in range(1, k // 2 + 1):
            # Connect to the next j nodes (with wraparound)
            initial_edges.append((i, (i + j) % num_nodes, {'weight': 1.0}))
    
    # Rewire edges
    for edge in list(initial_edges):  # Create a copy to iterate over
        if random.random() < beta:
            source, old_target, attrs = edge
            
            # Find a new target that is not the source and not already connected
            connected_to = set(target for s, target, _ in initial_edges if s == source)
            connected_to.add(source)  # Don't connect to self
            
            if len(connected_to) < num_nodes:  # Only rewire if there are available targets
                while True:
                    new_target = random.randint(0, num_nodes - 1)
                    if new_target not in connected_to:
                        break
                
                # Remove the old edge and add the new one
                initial_edges.remove(edge)
                initial_edges.append((source, new_target, attrs))
    
    # Create the simulation
    return NetworkSimulation(
        initial_nodes=initial_nodes,
        initial_edges=initial_edges,
        directed=directed
    )