"""Epidemiological Simulation implementation."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("Epidemiological")
class EpidemiologicalSimulation(BaseSimulation):
    """A simulation class for epidemiological models (SIR, SEIR, etc.).
    
    This implementation focuses on the classic SIR (Susceptible, Infected, Recovered)
    model for disease spread in a population.
    
    Attributes:
        population_size (int): Total population size.
        initial_infected (int): Initial number of infected individuals.
        initial_recovered (int): Initial number of recovered individuals.
        beta (float): Transmission rate (rate at which susceptible individuals become infected).
        gamma (float): Recovery rate (rate at which infected individuals recover).
        days (int): Number of days to simulate.
        random_seed (Optional[int]): Seed for random number generation.
    """
    
    def __init__(
        self, 
        population_size: int,
        initial_infected: int,
        initial_recovered: int = 0,
        beta: float = 0.3,
        gamma: float = 0.1,
        days: int = 100,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the epidemiological simulation.
        
        Args:
            population_size: Total number of individuals in the population.
            initial_infected: Initial number of infected individuals.
            initial_recovered: Initial number of recovered individuals.
            beta: Transmission rate (rate at which susceptible individuals become infected).
            gamma: Recovery rate (rate at which infected individuals recover).
            days: Number of days to simulate.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        
        # Validate inputs
        if initial_infected + initial_recovered > population_size:
            raise ValueError("Initial infected + recovered cannot exceed population size")
        
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.beta = beta
        self.gamma = gamma
        
        # Calculate initial susceptible
        self.initial_susceptible = population_size - initial_infected - initial_recovered
        
        # State variables to store simulation results
        self.susceptible = []
        self.infected = []
        self.recovered = []
    
    def run_simulation(self) -> List[float]:
        """Run the SIR epidemiological model simulation.
        
        Returns:
            A list with the number of infected individuals each day.
        """
        self.reset()
        
        # Initialize compartments
        s = [self.initial_susceptible]
        i = [self.initial_infected]
        r = [self.initial_recovered]
        
        # Run the SIR model for each day
        for _ in range(1, self.days):
            # Current values
            s_current = s[-1]
            i_current = i[-1]
            r_current = r[-1]
            
            # Calculate new infections and recoveries
            new_infections = self.beta * s_current * i_current / self.population_size
            new_recoveries = self.gamma * i_current
            
            # Update compartments
            s_next = s_current - new_infections
            i_next = i_current + new_infections - new_recoveries
            r_next = r_current + new_recoveries
            
            # Ensure values don't go below zero (floating point errors)
            s_next = max(0, s_next)
            i_next = max(0, i_next)
            
            # Store results
            s.append(s_next)
            i.append(i_next)
            r.append(r_next)
        
        # Store results for later access
        self.susceptible = s
        self.infected = i
        self.recovered = r
        
        # Return infected counts (main result)
        return i
    
    def get_compartments(self) -> Dict[str, List[float]]:
        """Get the full SIR compartment data for the simulation.
        
        Returns:
            A dictionary with lists for each compartment (susceptible, infected, recovered).
        """
        if not self.susceptible or not self.infected or not self.recovered:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        return {
            "susceptible": self.susceptible,
            "infected": self.infected,
            "recovered": self.recovered
        }
    
    def get_peak_infection(self) -> Tuple[int, float]:
        """Get the day and value of peak infection.
        
        Returns:
            A tuple of (day, peak_value) for the maximum number of infected individuals.
        """
        if not self.infected:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        peak_value = max(self.infected)
        peak_day = self.infected.index(peak_value)
        
        return peak_day, peak_value
    
    def get_reproduction_number(self) -> float:
        """Calculate the basic reproduction number (R0).
        
        The basic reproduction number is the expected number of cases directly generated
        by one case in a population where all individuals are susceptible.
        
        Returns:
            The basic reproduction number (R0 = beta/gamma).
        """
        return self.beta / self.gamma
    
    def get_final_sizes(self) -> Dict[str, float]:
        """Get the final sizes of each compartment.
        
        Returns:
            A dictionary with the final number of individuals in each compartment.
        """
        if not self.susceptible or not self.infected or not self.recovered:
            raise ValueError("No simulation results available. Run the simulation first.")
        
        return {
            "susceptible": self.susceptible[-1],
            "infected": self.infected[-1],
            "recovered": self.recovered[-1]
        }
    
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
            'population_size': {
                'type': 'int',
                'description': 'Total number of individuals in the population',
                'required': True
            },
            'initial_infected': {
                'type': 'int',
                'description': 'Initial number of infected individuals',
                'required': True
            },
            'initial_recovered': {
                'type': 'int',
                'description': 'Initial number of recovered individuals',
                'required': False,
                'default': 0
            },
            'beta': {
                'type': 'float',
                'description': 'Transmission rate (rate at which susceptible individuals become infected)',
                'required': False,
                'default': 0.3
            },
            'gamma': {
                'type': 'float',
                'description': 'Recovery rate (rate at which infected individuals recover)',
                'required': False,
                'default': 0.1
            }
        })
        
        return params