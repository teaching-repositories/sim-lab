import numpy as np
from typing import Any, Dict, List, Optional

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("ResourceFluctuations")
class ResourceFluctuationsSimulation(BaseSimulation):
    """
    A simulation class to model the fluctuations of resource prices over time,
    considering factors like volatility, market trends (drift), and supply disruptions.

    Attributes:
        start_price (float): The initial price of the resource.
        days (int): The duration of the simulation in days.
        volatility (float): The volatility of price changes, representing day-to-day variability.
        drift (float): The average daily price change, indicating the trend over time.
        supply_disruption_day (Optional[int]): The specific day a supply disruption occurs (default is None).
        disruption_severity (float): The magnitude of the disruption's impact on price (default is 0).
        random_seed (Optional[int]): The seed for the random number generator to ensure reproducibility (default is None).
    """
    def __init__(self, start_price: float, days: int, volatility: float, drift: float,
                 supply_disruption_day: Optional[int] = None, disruption_severity: float = 0,
                 random_seed: Optional[int] = None) -> None:
        """
        Initializes the ResourceSimulation with all necessary parameters.

        Parameters:
            start_price (float): The initial price of the resource.
            days (int): The total number of days to simulate.
            volatility (float): The volatility of the resource price, representing the randomness of day-to-day price changes.
            drift (float): The expected daily percentage change in price, which can be positive or negative.
            supply_disruption_day (Optional[int]): Day on which a supply disruption occurs (defaults to None).
            disruption_severity (float): The severity of the supply disruption, affecting prices multiplicatively.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducible results (defaults to None).
        """
        super().__init__(days=days, random_seed=random_seed)
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift
        self.supply_disruption_day = supply_disruption_day
        self.disruption_severity = disruption_severity

    def run_simulation(self) -> List[float]:
        """
        Simulates the price of the resource over a specified number of days based on the initial settings.

        Returns:
            List[float]: A list containing the price of the resource for each day of the simulation.
        """
        # Base class handles random seed initialization
        self.reset()
        
        prices = [self.start_price]
        for day in range(1, self.days):
            previous_price = prices[-1]
            random_change = np.random.normal(self.drift, self.volatility)
            new_price = previous_price * (1 + random_change)

            if day == self.supply_disruption_day:
                new_price = previous_price * (1 + self.disruption_severity)

            prices.append(new_price)

        return prices
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata (type, description, default, etc.)
        """
        # Get base parameters from parent class
        params = super().get_parameters_info()
        
        # Add class-specific parameters
        params.update({
            'start_price': {
                'type': 'float',
                'description': 'The initial price of the resource',
                'required': True
            },
            'volatility': {
                'type': 'float',
                'description': 'The volatility of resource price changes, representing day-to-day variability',
                'required': True
            },
            'drift': {
                'type': 'float',
                'description': 'The average daily price change, indicating the trend over time',
                'required': True
            },
            'supply_disruption_day': {
                'type': 'int',
                'description': 'The specific day a supply disruption occurs',
                'required': False,
                'default': None
            },
            'disruption_severity': {
                'type': 'float',
                'description': 'The magnitude of the disruption\'s impact on prices',
                'required': False,
                'default': 0
            }
        })
        
        return params
