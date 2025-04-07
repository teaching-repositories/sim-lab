"""Markov Chain Simulation implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("MarkovChain")
class MarkovChainSimulation(BaseSimulation):
    """A simulation class for Markov chain processes.
    
    This simulation models stochastic processes where the future state depends only on the current state,
    not on the sequence of events that preceded it (the Markov property).
    
    Attributes:
        transition_matrix (np.ndarray): Matrix of transition probabilities between states.
        states (List[Any]): List of possible states.
        initial_state (int): Index of the initial state.
        current_state (int): Index of the current state.
        state_history (List[int]): History of state indices.
        days (int): Number of steps to simulate.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self,
        transition_matrix: np.ndarray,
        states: Optional[List[Any]] = None,
        initial_state: Optional[Union[int, Any]] = None,
        days: int = 100,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the Markov chain simulation.
        
        Args:
            transition_matrix: Square matrix of transition probabilities. Each row must sum to 1.
            states: List of state names or values. If None, states will be numbered [0, 1, 2, ...].
            initial_state: Index or name of the initial state. If None, a random state is selected.
            days: Number of steps to simulate.
            random_seed: Seed for random number generation.
            
        Raises:
            ValueError: If the transition matrix is invalid.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        # Validate the transition matrix
        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)
        
        if len(transition_matrix.shape) != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
        
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Each row of the transition matrix must sum to 1")
        
        self.transition_matrix = transition_matrix
        self.num_states = transition_matrix.shape[0]
        
        # Set up states
        if states is None:
            self.states = list(range(self.num_states))
        else:
            if len(states) != self.num_states:
                raise ValueError("Number of states must match the dimension of the transition matrix")
            self.states = states
        
        # Set up state mapping
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        
        # Set initial state
        if initial_state is None:
            self.initial_state = np.random.randint(0, self.num_states)
        elif isinstance(initial_state, int):
            if initial_state < 0 or initial_state >= self.num_states:
                raise ValueError(f"Initial state index {initial_state} is out of range")
            self.initial_state = initial_state
        else:
            if initial_state not in self.state_to_index:
                raise ValueError(f"Initial state {initial_state} not found in state list")
            self.initial_state = self.state_to_index[initial_state]
        
        self.current_state = self.initial_state
        self.state_history = [self.initial_state]
        
        # Additional properties
        self.stationary_distribution = None
        
    def step(self) -> int:
        """Take one step in the Markov chain.
        
        Returns:
            The index of the new state.
        """
        # Sample the next state based on the current state's transition probabilities
        next_state = np.random.choice(
            self.num_states,
            p=self.transition_matrix[self.current_state]
        )
        
        # Update the current state and history
        self.current_state = next_state
        self.state_history.append(next_state)
        
        return next_state
    
    def run_simulation(self) -> List[int]:
        """Run the Markov chain simulation.
        
        Returns:
            A list of state indices for each time step.
        """
        self.reset()
        
        # Run for specified number of days (steps)
        for _ in range(1, self.days):
            self.step()
        
        return self.state_history
    
    def get_state_names(self) -> List[Any]:
        """Get the state names corresponding to the state indices.
        
        Returns:
            A list of state names for each time step.
        """
        return [self.states[i] for i in self.state_history]
    
    def get_state_distribution(self) -> Dict[Any, float]:
        """Get the distribution of states in the simulation.
        
        Returns:
            A dictionary mapping state names to their frequency.
        """
        # Count occurrences of each state
        state_counts = np.zeros(self.num_states)
        for state in self.state_history:
            state_counts[state] += 1
        
        # Convert to frequencies
        state_freq = state_counts / len(self.state_history)
        
        # Return as a dictionary
        return {self.states[i]: freq for i, freq in enumerate(state_freq)}
    
    def compute_stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution of the Markov chain.
        
        The stationary distribution is the long-run proportion of time spent in each state.
        
        Returns:
            A NumPy array of probabilities for each state.
        """
        # For a regular Markov chain, the stationary distribution is the eigenvector of
        # the transition matrix with eigenvalue 1, normalized to sum to 1.
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find the eigenvector corresponding to eigenvalue 1
        for i, eigenvalue in enumerate(eigenvalues):
            if np.isclose(eigenvalue, 1.0):
                # Normalize the eigenvector
                stationary = np.real(eigenvectors[:, i])
                stationary = stationary / np.sum(stationary)
                self.stationary_distribution = stationary
                return stationary
        
        # If no eigenvalue is 1, the chain may not have a stationary distribution
        raise ValueError("Could not compute stationary distribution")
    
    def predict_state_probabilities(self, steps: int) -> np.ndarray:
        """Predict the probabilities of being in each state after a given number of steps.
        
        Args:
            steps: Number of steps into the future.
            
        Returns:
            A NumPy array of probabilities for each state.
        """
        # Create a probability vector with 1 at the current state
        prob_vector = np.zeros(self.num_states)
        prob_vector[self.current_state] = 1.0
        
        # Multiply by the transition matrix 'steps' times
        for _ in range(steps):
            prob_vector = prob_vector @ self.transition_matrix
        
        return prob_vector
    
    def get_most_likely_state_sequence(self, observations: List[Any]) -> List[int]:
        """Get the most likely sequence of hidden states given observations (Viterbi algorithm).
        
        This method is intended for hidden Markov models (HMMs) where the observations are 
        probabilistically related to the hidden states.
        
        Args:
            observations: List of observed values.
            
        Returns:
            A list of state indices representing the most likely state sequence.
            
        Raises:
            ValueError: If the simulation is not a hidden Markov model.
        """
        # This is a placeholder for the Viterbi algorithm
        # For a full HMM implementation, emission probabilities would be required
        raise NotImplementedError("Hidden Markov Model functionality not implemented")
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        self.current_state = self.initial_state
        self.state_history = [self.initial_state]
    
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
            'transition_matrix': {
                'type': 'np.ndarray',
                'description': 'Square matrix of transition probabilities. Each row must sum to 1.',
                'required': True
            },
            'states': {
                'type': 'List[Any]',
                'description': 'List of state names or values. If None, states will be numbered [0, 1, 2, ...].',
                'required': False,
                'default': None
            },
            'initial_state': {
                'type': 'Union[int, Any]',
                'description': 'Index or name of the initial state. If None, a random state is selected.',
                'required': False,
                'default': None
            }
        })
        
        return params


# Common Markov chain models
def create_weather_model(
    sunny_to_sunny: float = 0.7,
    sunny_to_cloudy: float = 0.2,
    sunny_to_rainy: float = 0.1,
    cloudy_to_sunny: float = 0.3,
    cloudy_to_cloudy: float = 0.4,
    cloudy_to_rainy: float = 0.3,
    rainy_to_sunny: float = 0.2,
    rainy_to_cloudy: float = 0.3,
    rainy_to_rainy: float = 0.5,
    initial_state: str = "Sunny",
    days: int = 30
) -> MarkovChainSimulation:
    """Create a simple weather model as a Markov chain.
    
    Args:
        sunny_to_sunny: Probability of sunny day followed by sunny day.
        sunny_to_cloudy: Probability of sunny day followed by cloudy day.
        sunny_to_rainy: Probability of sunny day followed by rainy day.
        cloudy_to_sunny: Probability of cloudy day followed by sunny day.
        cloudy_to_cloudy: Probability of cloudy day followed by cloudy day.
        cloudy_to_rainy: Probability of cloudy day followed by rainy day.
        rainy_to_sunny: Probability of rainy day followed by sunny day.
        rainy_to_cloudy: Probability of rainy day followed by cloudy day.
        rainy_to_rainy: Probability of rainy day followed by rainy day.
        initial_state: The initial weather state.
        days: Number of days to simulate.
        
    Returns:
        A MarkovChainSimulation configured as a weather model.
    """
    # Define the transition matrix
    transition_matrix = np.array([
        [sunny_to_sunny, sunny_to_cloudy, sunny_to_rainy],
        [cloudy_to_sunny, cloudy_to_cloudy, cloudy_to_rainy],
        [rainy_to_sunny, rainy_to_cloudy, rainy_to_rainy]
    ])
    
    # Define the states
    states = ["Sunny", "Cloudy", "Rainy"]
    
    # Create the simulation
    return MarkovChainSimulation(
        transition_matrix=transition_matrix,
        states=states,
        initial_state=initial_state,
        days=days
    )


def create_random_walk(
    p_up: float = 0.5,
    p_down: float = 0.5,
    initial_position: int = 0,
    min_position: int = -10,
    max_position: int = 10,
    days: int = 100
) -> MarkovChainSimulation:
    """Create a random walk model as a Markov chain.
    
    Args:
        p_up: Probability of moving up.
        p_down: Probability of moving down.
        initial_position: The initial position.
        min_position: The minimum allowed position.
        max_position: The maximum allowed position.
        days: Number of steps to simulate.
        
    Returns:
        A MarkovChainSimulation configured as a random walk.
    """
    # Adjust probabilities to sum to 1
    total = p_up + p_down
    p_up = p_up / total
    p_down = p_down / total
    
    # Number of states
    num_states = max_position - min_position + 1
    
    # Define the transition matrix
    transition_matrix = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        position = min_position + i
        
        # Boundary conditions
        if position == min_position:
            transition_matrix[i, i + 1] = 1.0  # Can only go up
        elif position == max_position:
            transition_matrix[i, i - 1] = 1.0  # Can only go down
        else:
            # Normal case: can go up or down
            transition_matrix[i, i + 1] = p_up
            transition_matrix[i, i - 1] = p_down
    
    # Define the states
    states = list(range(min_position, max_position + 1))
    
    # Map initial position to state index
    initial_state = initial_position - min_position
    
    # Create the simulation
    return MarkovChainSimulation(
        transition_matrix=transition_matrix,
        states=states,
        initial_state=initial_state,
        days=days
    )
    

def create_inventory_model(
    demand_probs: List[float] = [0.2, 0.3, 0.3, 0.2],
    max_inventory: int = 5,
    order_amount: int = 3,
    days: int = 30
) -> MarkovChainSimulation:
    """Create an inventory model as a Markov chain.
    
    The inventory level is the state, and demand follows a probability distribution.
    
    Args:
        demand_probs: Probability of each demand level (0, 1, 2, ...).
        max_inventory: Maximum inventory level.
        order_amount: Amount to order when inventory reaches 0.
        days: Number of days to simulate.
        
    Returns:
        A MarkovChainSimulation configured as an inventory model.
    """
    # Number of states (inventory levels from 0 to max_inventory)
    num_states = max_inventory + 1
    
    # Define the transition matrix
    transition_matrix = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        inventory = i
        
        for demand, prob in enumerate(demand_probs):
            # New inventory after demand
            new_inventory = max(0, inventory - demand)
            
            # Reorder if inventory is 0
            if new_inventory == 0:
                new_inventory = min(max_inventory, order_amount)
            
            # Update transition probability
            transition_matrix[i, new_inventory] += prob
    
    # Define the states
    states = list(range(num_states))
    
    # Create the simulation
    return MarkovChainSimulation(
        transition_matrix=transition_matrix,
        states=states,
        initial_state=max_inventory,  # Start with full inventory
        days=days
    )