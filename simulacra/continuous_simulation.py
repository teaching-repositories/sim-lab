from simulacra.base_simulation import BaseSimulation
import numpy as np
from typing import List
from random import normalvariate

class ContinuousSimulation(BaseSimulation):
    def __init__(self, start_value: float, timesteps: int, volatility: float, 
                 drift: float, time_unit: str = "days") -> None:
        super().__init__(timesteps, time_unit=time_unit)
        self.start_value: float = start_value
        self.volatility: float = volatility
        self.drift: float = drift

    def generate_values(self) -> List[float]:  
        values: List[float] = [self.start_value]
        for _ in range(1, self.timesteps):
            change: float = normalvariate(mu=self.drift, sigma=self.volatility)
            new_value: float = values[-1] * (1 + change)
            values.append(new_value)
        return values

    def run_simulation(self) -> np.ndarray: 
        return np.array(self.generate_values()) 
