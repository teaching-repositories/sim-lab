"""Agent-Based Simulation implementation."""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


class Agent:
    """Base class for all agents in an agent-based simulation.
    
    This class defines the common interface for all agents, including
    methods for updating state and interacting with other agents.
    
    Attributes:
        state (Dict[str, Any]): The current state of the agent.
        position (Optional[Tuple[float, float]]): The agent's position, if applicable.
        agent_id (int): Unique identifier for the agent.
    """
    
    def __init__(
        self, 
        agent_id: int,
        initial_state: Dict[str, Any] = None,
        position: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            initial_state: Initial state of the agent.
            position: Initial position of the agent, if applicable.
        """
        self.agent_id = agent_id
        self.state = initial_state or {}
        self.position = position
        self.history = []  # Track state history if needed
    
    def update(self, environment: Any, neighbors: List['Agent']) -> None:
        """Update the agent's state based on the environment and neighbors.
        
        This method should be implemented by subclasses to define agent behavior.
        
        Args:
            environment: The environment state or model information.
            neighbors: List of neighboring agents that can influence this agent.
        """
        pass
    
    def move(self, new_position: Tuple[float, float]) -> None:
        """Move the agent to a new position.
        
        Args:
            new_position: The new (x, y) position for the agent.
        """
        self.position = new_position
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent.
        
        Returns:
            The current state dictionary.
        """
        return self.state
    
    def save_history(self) -> None:
        """Save the current state to the agent's history."""
        self.history.append(self.state.copy())
    
    def reset(self) -> None:
        """Reset the agent's history."""
        self.history = []


class Environment:
    """Represents the environment in which agents operate.
    
    The environment contains global state information and may provide
    methods for agents to interact with it.
    
    Attributes:
        state (Dict[str, Any]): The current state of the environment.
        bounds (Tuple[float, float, float, float]): The spatial bounds of the environment (x_min, y_min, x_max, y_max).
    """
    
    def __init__(
        self,
        initial_state: Dict[str, Any] = None,
        bounds: Tuple[float, float, float, float] = (0, 0, 100, 100)
    ) -> None:
        """Initialize a new environment.
        
        Args:
            initial_state: Initial state of the environment.
            bounds: The spatial bounds as (x_min, y_min, x_max, y_max).
        """
        self.state = initial_state or {}
        self.bounds = bounds
        self.history = []
    
    def update(self, agents: List[Agent]) -> None:
        """Update the environment based on agent states.
        
        Args:
            agents: List of all agents in the simulation.
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment.
        
        Returns:
            The current state dictionary.
        """
        return self.state
    
    def save_history(self) -> None:
        """Save the current state to the environment's history."""
        self.history.append(self.state.copy())
    
    def reset(self) -> None:
        """Reset the environment's history."""
        self.history = []


@SimulatorRegistry.register("AgentBased")
class AgentBasedSimulation(BaseSimulation):
    """A simulation class for agent-based modeling.
    
    This simulation models complex systems by simulating the actions and interactions
    of autonomous agents, allowing for emergent behavior to be observed.
    
    Attributes:
        agents (List[Agent]): List of agents in the simulation.
        environment (Environment): The environment in which agents operate.
        days (int): Number of steps to simulate.
        neighborhood_radius (float): Radius for determining agent neighbors.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self,
        agent_factory: Callable[[int], Agent],
        num_agents: int,
        environment: Optional[Environment] = None,
        days: int = 100,
        neighborhood_radius: float = 10.0,
        save_history: bool = False,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the agent-based simulation.
        
        Args:
            agent_factory: Function that creates new agents with given IDs.
            num_agents: Number of agents to create.
            environment: The environment in which agents operate. If None, a default environment is created.
            days: Number of steps to simulate.
            neighborhood_radius: Radius for determining agent neighbors.
            save_history: Whether to save agent and environment history.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        self.environment = environment or Environment()
        self.neighborhood_radius = neighborhood_radius
        self.save_history = save_history
        
        # Create agents
        self.agents = [agent_factory(i) for i in range(num_agents)]
        
        # Initialize tracking variables
        self.metrics = []
    
    def get_agent_neighbors(self, agent: Agent) -> List[Agent]:
        """Get the neighbors of an agent based on proximity.
        
        Args:
            agent: The agent whose neighbors to find.
            
        Returns:
            List of neighboring agents within the neighborhood radius.
        """
        if agent.position is None:
            return []  # No position, no neighbors
        
        neighbors = []
        for other in self.agents:
            if other.agent_id == agent.agent_id or other.position is None:
                continue  # Skip self and agents without position
                
            # Calculate Euclidean distance
            dx = agent.position[0] - other.position[0]
            dy = agent.position[1] - other.position[1]
            distance = (dx**2 + dy**2)**0.5
            
            if distance <= self.neighborhood_radius:
                neighbors.append(other)
                
        return neighbors
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current simulation state.
        
        Override this method to define specific metrics for your simulation.
        
        Returns:
            Dictionary of metrics derived from agent and environment states.
        """
        # Default implementation: count agents in different states
        state_counts = {}
        
        for agent in self.agents:
            for key, value in agent.state.items():
                if isinstance(value, (bool, int, str, float)):
                    state_key = f"{key}_{value}"
                    state_counts[state_key] = state_counts.get(state_key, 0) + 1
        
        return state_counts
    
    def run_simulation(self) -> List[Dict[str, Any]]:
        """Run the agent-based simulation.
        
        Returns:
            A list of metrics dictionaries for each time step.
        """
        self.reset()
        
        # Initialize history if tracking
        if self.save_history:
            for agent in self.agents:
                agent.reset()
                agent.save_history()
            self.environment.reset()
            self.environment.save_history()
        
        # Calculate initial metrics
        self.metrics = [self.calculate_metrics()]
        
        # Run for specified number of days
        for _ in range(1, self.days):
            # Update agents
            for agent in self.agents:
                neighbors = self.get_agent_neighbors(agent)
                agent.update(self.environment, neighbors)
                
                if self.save_history:
                    agent.save_history()
            
            # Update environment
            self.environment.update(self.agents)
            
            if self.save_history:
                self.environment.save_history()
            
            # Calculate metrics
            self.metrics.append(self.calculate_metrics())
        
        return self.metrics
    
    def get_agent_history(self, agent_id: int) -> List[Dict[str, Any]]:
        """Get the state history for a specific agent.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            List of state dictionaries representing the agent's history.
        """
        if not self.save_history:
            raise ValueError("Agent history was not saved. Set save_history=True when creating the simulation.")
        
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.history
        
        raise ValueError(f"No agent with ID {agent_id} found")
    
    def get_environment_history(self) -> List[Dict[str, Any]]:
        """Get the environment state history.
        
        Returns:
            List of state dictionaries representing the environment's history.
        """
        if not self.save_history:
            raise ValueError("Environment history was not saved. Set save_history=True when creating the simulation.")
        
        return self.environment.history
    
    def get_metric_history(self, metric_name: str) -> List[Any]:
        """Get the history of a specific metric.
        
        Args:
            metric_name: The name of the metric to retrieve.
            
        Returns:
            List of values for the specified metric over time.
        """
        if not self.metrics:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        try:
            return [metrics[metric_name] for metrics in self.metrics]
        except KeyError:
            raise ValueError(f"Metric '{metric_name}' not found in simulation results")
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        self.metrics = []
    
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
            'agent_factory': {
                'type': 'Callable[[int], Agent]',
                'description': 'Function that creates new agents with given IDs',
                'required': True
            },
            'num_agents': {
                'type': 'int',
                'description': 'Number of agents to create',
                'required': True
            },
            'environment': {
                'type': 'Environment',
                'description': 'The environment in which agents operate',
                'required': False,
                'default': 'Default Environment'
            },
            'neighborhood_radius': {
                'type': 'float',
                'description': 'Radius for determining agent neighbors',
                'required': False,
                'default': 10.0
            },
            'save_history': {
                'type': 'bool',
                'description': 'Whether to save agent and environment history',
                'required': False,
                'default': False
            }
        })
        
        return params