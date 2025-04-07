import numpy as np
from typing import Any, Dict, List, Optional

from .base_simulation import BaseSimulation
from .registry import SimulatorRegistry


@SimulatorRegistry.register("StockMarket")
class StockMarketSimulation(BaseSimulation):
    """
    A simulation class to model the fluctuations of stock prices over time, accounting for volatility,
    general market trends (drift), and specific market events.

    Attributes:
        start_price (float): The initial price of the stock.
        days (int): The duration of the simulation in days.
        volatility (float): The volatility of stock price changes, representing day-to-day variability.
        drift (float): The average daily price change, indicating the trend over time.
        event_day (Optional[int]): The specific day a major market event occurs (default is None).
        event_impact (float): The magnitude of the event's impact on stock prices (default is 0).
        random_seed (Optional[int]): The seed for the random number generator to ensure reproducibility (default is None).

    Methods:
        run_simulation(): Runs the simulation and returns a list of stock prices over the simulation period.
    """

    def __init__(
        self, start_price: float, days: int, volatility: float, drift: float,
        event_day: Optional[int] = None, event_impact: float = 0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initializes the StockMarketSimulation with all necessary parameters.

        Parameters:
            start_price (float): The initial stock price.
            days (int): The total number of days to simulate.
            volatility (float): The volatility of the stock price, representing the randomness of day-to-day price changes.
            drift (float): The expected daily percentage change in price, which can be positive or negative.
            event_day (Optional[int]): Day on which a major market event occurs (defaults to None).
            event_impact (float): The severity of the market event, affecting prices multiplicatively.
            random_seed (Optional[int]): Seed for the random number generator to ensure reproducible results (defaults to None).
        """
        super().__init__(days=days, random_seed=random_seed)
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift
        self.event_day = event_day
        self.event_impact = event_impact

    def run_simulation(self) -> List[float]:
        """
        Simulates the stock price over a specified number of days based on the initial settings.

        Returns:
            List[float]: A list containing the stock prices for each day of the simulation.
        """
        # Base class handles random seed initialization
        self.reset()
        
        prices = [self.start_price]
        for day in range(1, self.days):
            previous_price = prices[-1]
            random_change = np.random.normal(self.drift, self.volatility)
            new_price = previous_price * (1 + random_change)

            if day == self.event_day:
                new_price = previous_price * (1 + self.event_impact)

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
                'description': 'The initial price of the stock',
                'required': True
            },
            'volatility': {
                'type': 'float',
                'description': 'The volatility of stock price changes, representing day-to-day variability',
                'required': True
            },
            'drift': {
                'type': 'float',
                'description': 'The average daily price change, indicating the trend over time',
                'required': True
            },
            'event_day': {
                'type': 'int',
                'description': 'The specific day a major market event occurs',
                'required': False,
                'default': None
            },
            'event_impact': {
                'type': 'float',
                'description': 'The magnitude of the event\'s impact on stock prices',
                'required': False,
                'default': 0
            }
        })
        
        return params
