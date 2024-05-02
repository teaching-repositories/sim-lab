from simulacra.base_simulation import BaseSimulation
import numpy as np
from typing import List
from random import choice

class DiscreteSimulation(BaseSimulation):
    def __init__(self, start_value: float, timesteps: int, possible_values: List[float], probabilities: List[float], time_unit: str = "days") -> None:
        super().__init__(timesteps, time_unit=time_unit)
        self.start_value: float = start_value
        self.possible_values: List[float] = possible_values
        self.probabilities: List[float] = probabilities

        # Check if the lengths of possible_values and probabilities match
        if len(self.possible_values) != len(self.probabilities):
            raise ValueError("Lengths of possible_values and probabilities must match")

    def generate_values(self) -> List[float]:
        values: List[float] = [self.start_value]
        for _ in range(1, self.timesteps):
            # Randomly choose a value based on the given probabilities
            new_value = choice(self.possible_values, p=self.probabilities)
            values.append(new_value)
        return values

    def run_simulation(self) -> np.ndarray:
        return np.array(self.generate_values())
