from .continuous_simulation import ContinuousSimulation
from typing import Optional


class StockMarketSimulation(ContinuousSimulation):
    """
    A simulation class to model stock market behaviour with specified initial conditions and market dynamics.
    """

    def __init__(self, start_price: float, days: int, volatility: float, drift: float, event_day: Optional[int] = None, event_impact: Optional[float] = None) -> None:
        """
        Initialises a new stock market simulation.

        Parameters:
            start_price (float): The starting price of the stock.
            days (int): The total number of days to simulate.
            volatility (float): The volatility of the stock price, reflecting price fluctuation.
            drift (float): The drift of the stock price, reflecting a trend over time.
            event_day (Optional[int]): A specific day an event occurs that might affect the stock price.
            event_impact (Optional[float]): The impact of the event on the stock price.
        """
        # super().__init__(days)
        pass
