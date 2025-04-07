"""Predator-Prey Ecosystem Simulation implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("PredatorPrey")
class PredatorPreySimulation(BaseSimulation):
    """A simulation class for predator-prey ecosystem dynamics.
    
    This simulation models the interactions between predator and prey populations
    based on the Lotka-Volterra equations and extensions.
    
    Attributes:
        prey_population (float): Current prey population size.
        predator_population (float): Current predator population size.
        prey_growth_rate (float): Natural growth rate of prey (without predators).
        predation_rate (float): Rate at which predators consume prey.
        predator_death_rate (float): Natural death rate of predators (without prey).
        predator_growth_factor (float): Conversion rate of prey into new predators.
        carrying_capacity (Optional[float]): Environmental carrying capacity for prey.
        competition_factor (Optional[float]): Intraspecies competition factor.
        days (int): Number of days to simulate.
        dt (float): Time step size for numerical integration.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self,
        initial_prey: float,
        initial_predators: float,
        prey_growth_rate: float,
        predation_rate: float,
        predator_death_rate: float,
        predator_growth_factor: float,
        carrying_capacity: Optional[float] = None,
        competition_factor: Optional[float] = None,
        days: int = 100,
        dt: float = 0.1,
        stochastic: bool = False,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the predator-prey simulation.
        
        Args:
            initial_prey: Initial prey population size.
            initial_predators: Initial predator population size.
            prey_growth_rate: Natural growth rate of prey (without predators).
            predation_rate: Rate at which predators consume prey.
            predator_death_rate: Natural death rate of predators (without prey).
            predator_growth_factor: Conversion rate of prey into new predators.
            carrying_capacity: Environmental carrying capacity for prey (for logistic growth).
            competition_factor: Intraspecies competition factor.
            days: Number of days to simulate.
            dt: Time step size for numerical integration.
            stochastic: Whether to add random noise to the simulation.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        # Validate inputs
        if initial_prey < 0 or initial_predators < 0:
            raise ValueError("Initial populations must be non-negative")
        
        if prey_growth_rate < 0 or predation_rate < 0 or predator_death_rate < 0 or predator_growth_factor < 0:
            raise ValueError("Rates must be non-negative")
        
        if carrying_capacity is not None and carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be positive")
        
        # Set parameters
        self.initial_prey = initial_prey
        self.initial_predators = initial_predators
        self.prey_growth_rate = prey_growth_rate
        self.predation_rate = predation_rate
        self.predator_death_rate = predator_death_rate
        self.predator_growth_factor = predator_growth_factor
        self.carrying_capacity = carrying_capacity
        self.competition_factor = competition_factor
        self.dt = dt
        self.stochastic = stochastic
        
        # Initialize populations
        self.prey_population = initial_prey
        self.predator_population = initial_predators
        
        # Initialize history
        self.prey_history = [initial_prey]
        self.predator_history = [initial_predators]
    
    def get_derivatives(
        self, 
        prey: float, 
        predators: float
    ) -> Tuple[float, float]:
        """Calculate the rate of change of prey and predator populations.
        
        Args:
            prey: Current prey population.
            predators: Current predator population.
            
        Returns:
            A tuple of (prey_derivative, predator_derivative).
        """
        # Basic Lotka-Volterra model
        prey_derivative = prey * self.prey_growth_rate - prey * predators * self.predation_rate
        predator_derivative = predators * self.predator_growth_factor * prey - predators * self.predator_death_rate
        
        # Add carrying capacity (logistic growth) if specified
        if self.carrying_capacity is not None:
            prey_derivative -= prey * self.prey_growth_rate * (prey / self.carrying_capacity)
        
        # Add competition factor if specified
        if self.competition_factor is not None:
            prey_derivative -= prey**2 * self.competition_factor
            predator_derivative -= predators**2 * self.competition_factor
        
        return prey_derivative, predator_derivative
    
    def step(self) -> Tuple[float, float]:
        """Take one time step in the simulation.
        
        Returns:
            The new prey and predator populations as a tuple.
        """
        # Current values
        prey = self.prey_population
        predators = self.predator_population
        
        # Calculate derivatives
        prey_derivative, predator_derivative = self.get_derivatives(prey, predators)
        
        # Add stochastic element if enabled
        if self.stochastic:
            prey_noise = np.random.normal(0, 0.05 * prey) if prey > 0 else 0
            predator_noise = np.random.normal(0, 0.05 * predators) if predators > 0 else 0
            prey_derivative += prey_noise
            predator_derivative += predator_noise
        
        # Update populations using Euler method
        new_prey = max(0, prey + prey_derivative * self.dt)
        new_predators = max(0, predators + predator_derivative * self.dt)
        
        # Update state
        self.prey_population = new_prey
        self.predator_population = new_predators
        
        # Add to history
        self.prey_history.append(new_prey)
        self.predator_history.append(new_predators)
        
        return new_prey, new_predators
    
    def run_simulation(self) -> Dict[str, List[float]]:
        """Run the predator-prey simulation.
        
        Returns:
            A dictionary with lists of prey and predator population values over time.
        """
        self.reset()
        
        # Number of time steps
        num_steps = int(self.days / self.dt)
        
        # Run for the specified number of steps
        for _ in range(1, num_steps):
            self.step()
        
        # Return results at the desired time resolution (days)
        step_indices = np.linspace(0, len(self.prey_history) - 1, self.days, dtype=int)
        
        return {
            "prey": [self.prey_history[i] for i in step_indices],
            "predators": [self.predator_history[i] for i in step_indices]
        }
    
    def get_equilibrium_points(self) -> List[Tuple[float, float]]:
        """Calculate the equilibrium points of the predator-prey system.
        
        Equilibrium points are population levels where both derivatives are zero.
        
        Returns:
            A list of (prey, predator) equilibrium points.
        """
        equilibria = []
        
        # Trivial equilibrium at (0, 0)
        equilibria.append((0, 0))
        
        # Basic Lotka-Volterra equilibrium
        lotka_volterra_prey = self.predator_death_rate / self.predator_growth_factor
        lotka_volterra_predators = self.prey_growth_rate / self.predation_rate
        
        # Adjust for carrying capacity if specified
        if self.carrying_capacity is not None:
            if lotka_volterra_prey > self.carrying_capacity:
                # Only carrying capacity equilibrium is valid
                carrying_capacity_predators = self.prey_growth_rate * (1 - self.carrying_capacity / self.carrying_capacity) / self.predation_rate
                if carrying_capacity_predators >= 0:
                    equilibria.append((self.carrying_capacity, carrying_capacity_predators))
            else:
                # Both equilibria are valid
                if lotka_volterra_predators >= 0:
                    equilibria.append((lotka_volterra_prey, lotka_volterra_predators))
                
                carrying_capacity_predators = self.prey_growth_rate * (1 - self.carrying_capacity / self.carrying_capacity) / self.predation_rate
                if carrying_capacity_predators >= 0:
                    equilibria.append((self.carrying_capacity, carrying_capacity_predators))
        else:
            # Only Lotka-Volterra equilibrium
            if lotka_volterra_prey >= 0 and lotka_volterra_predators >= 0:
                equilibria.append((lotka_volterra_prey, lotka_volterra_predators))
        
        return equilibria
    
    def get_phase_diagram(self, num_points: int = 20) -> Dict[str, np.ndarray]:
        """Generate a phase diagram of the predator-prey system.
        
        The phase diagram shows the direction and magnitude of population changes at different
        population levels, which helps visualize the system dynamics.
        
        Args:
            num_points: Number of grid points in each dimension.
            
        Returns:
            A dictionary with grid coordinates and vector field components.
        """
        # Determine reasonable bounds based on history or equilibria
        if len(self.prey_history) > 1:
            max_prey = max(max(self.prey_history), self.initial_prey * 2)
            max_predators = max(max(self.predator_history), self.initial_predators * 2)
        else:
            # No history yet, use initial values and equilibria
            equilibria = self.get_equilibrium_points()
            max_prey = max([p[0] for p in equilibria] + [self.initial_prey]) * 2
            max_predators = max([p[1] for p in equilibria] + [self.initial_predators]) * 2
            
            # Ensure non-zero bounds
            max_prey = max(max_prey, 1.0)
            max_predators = max(max_predators, 1.0)
        
        # Create grid
        prey_grid = np.linspace(0, max_prey, num_points)
        predator_grid = np.linspace(0, max_predators, num_points)
        X, Y = np.meshgrid(prey_grid, predator_grid)
        
        # Calculate derivatives at each grid point
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(num_points):
            for j in range(num_points):
                prey_derivative, predator_derivative = self.get_derivatives(X[i, j], Y[i, j])
                U[i, j] = prey_derivative
                V[i, j] = predator_derivative
        
        return {
            "X": X,
            "Y": Y,
            "U": U,
            "V": V
        }
    
    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        super().reset()
        self.prey_population = self.initial_prey
        self.predator_population = self.initial_predators
        self.prey_history = [self.initial_prey]
        self.predator_history = [self.initial_predators]
    
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
            'initial_prey': {
                'type': 'float',
                'description': 'Initial prey population size',
                'required': True
            },
            'initial_predators': {
                'type': 'float',
                'description': 'Initial predator population size',
                'required': True
            },
            'prey_growth_rate': {
                'type': 'float',
                'description': 'Natural growth rate of prey (without predators)',
                'required': True
            },
            'predation_rate': {
                'type': 'float',
                'description': 'Rate at which predators consume prey',
                'required': True
            },
            'predator_death_rate': {
                'type': 'float',
                'description': 'Natural death rate of predators (without prey)',
                'required': True
            },
            'predator_growth_factor': {
                'type': 'float',
                'description': 'Conversion rate of prey into new predators',
                'required': True
            },
            'carrying_capacity': {
                'type': 'float',
                'description': 'Environmental carrying capacity for prey (for logistic growth)',
                'required': False,
                'default': None
            },
            'competition_factor': {
                'type': 'float',
                'description': 'Intraspecies competition factor',
                'required': False,
                'default': None
            },
            'dt': {
                'type': 'float',
                'description': 'Time step size for numerical integration',
                'required': False,
                'default': 0.1
            },
            'stochastic': {
                'type': 'bool',
                'description': 'Whether to add random noise to the simulation',
                'required': False,
                'default': False
            }
        })
        
        return params


def create_predator_prey_model(
    model_type: str = "basic",
    initial_prey: float = 100,
    initial_predators: float = 20,
    days: int = 100,
    **kwargs
) -> PredatorPreySimulation:
    """Create a predefined predator-prey ecosystem model.
    
    Args:
        model_type: Type of model to create ("basic", "logistic", "stochastic").
        initial_prey: Initial prey population size.
        initial_predators: Initial predator population size.
        days: Number of days to simulate.
        **kwargs: Additional parameters specific to each model type.
        
    Returns:
        A configured PredatorPreySimulation.
    """
    if model_type == "basic":
        # Basic Lotka-Volterra model
        prey_growth_rate = kwargs.get("prey_growth_rate", 0.1)
        predation_rate = kwargs.get("predation_rate", 0.01)
        predator_death_rate = kwargs.get("predator_death_rate", 0.05)
        predator_growth_factor = kwargs.get("predator_growth_factor", 0.005)
        
        return PredatorPreySimulation(
            initial_prey=initial_prey,
            initial_predators=initial_predators,
            prey_growth_rate=prey_growth_rate,
            predation_rate=predation_rate,
            predator_death_rate=predator_death_rate,
            predator_growth_factor=predator_growth_factor,
            days=days
        )
    
    elif model_type == "logistic":
        # Lotka-Volterra with logistic prey growth
        prey_growth_rate = kwargs.get("prey_growth_rate", 0.1)
        predation_rate = kwargs.get("predation_rate", 0.01)
        predator_death_rate = kwargs.get("predator_death_rate", 0.05)
        predator_growth_factor = kwargs.get("predator_growth_factor", 0.005)
        carrying_capacity = kwargs.get("carrying_capacity", 1000)
        
        return PredatorPreySimulation(
            initial_prey=initial_prey,
            initial_predators=initial_predators,
            prey_growth_rate=prey_growth_rate,
            predation_rate=predation_rate,
            predator_death_rate=predator_death_rate,
            predator_growth_factor=predator_growth_factor,
            carrying_capacity=carrying_capacity,
            days=days
        )
    
    elif model_type == "stochastic":
        # Stochastic Lotka-Volterra model
        prey_growth_rate = kwargs.get("prey_growth_rate", 0.1)
        predation_rate = kwargs.get("predation_rate", 0.01)
        predator_death_rate = kwargs.get("predator_death_rate", 0.05)
        predator_growth_factor = kwargs.get("predator_growth_factor", 0.005)
        
        return PredatorPreySimulation(
            initial_prey=initial_prey,
            initial_predators=initial_predators,
            prey_growth_rate=prey_growth_rate,
            predation_rate=predation_rate,
            predator_death_rate=predator_death_rate,
            predator_growth_factor=predator_growth_factor,
            stochastic=True,
            days=days
        )
    
    elif model_type == "competition":
        # Lotka-Volterra with intraspecies competition
        prey_growth_rate = kwargs.get("prey_growth_rate", 0.1)
        predation_rate = kwargs.get("predation_rate", 0.01)
        predator_death_rate = kwargs.get("predator_death_rate", 0.05)
        predator_growth_factor = kwargs.get("predator_growth_factor", 0.005)
        competition_factor = kwargs.get("competition_factor", 0.001)
        
        return PredatorPreySimulation(
            initial_prey=initial_prey,
            initial_predators=initial_predators,
            prey_growth_rate=prey_growth_rate,
            predation_rate=predation_rate,
            predator_death_rate=predator_death_rate,
            predator_growth_factor=predator_growth_factor,
            competition_factor=competition_factor,
            days=days
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")