"""
Example demonstrating the use of the simulator registry.

This script shows how to:
1. List available simulators
2. Create simulators using the registry
3. Run simulations using a common interface
4. Create a custom simulator and register it
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional

from sim_lab.core import BaseSimulation, SimulatorRegistry


def run_example():
    """Run a complete example of using the simulator registry."""
    # List all available simulators
    print("Available simulators:")
    for sim_name in SimulatorRegistry.list_simulators():
        print(f"  - {sim_name}")
    print()
    
    # Create and run stock market simulation
    print("Running Stock Market Simulation...")
    stock_sim = SimulatorRegistry.create(
        "StockMarket",
        start_price=100.0,
        days=100,
        volatility=0.02,
        drift=0.001,
        event_day=50,
        event_impact=0.05,
        random_seed=42
    )
    stock_results = stock_sim.run_simulation()
    
    # Create and run resource fluctuations simulation
    print("Running Resource Fluctuations Simulation...")
    resource_sim = SimulatorRegistry.create(
        "ResourceFluctuations",
        start_price=50.0,
        days=100,
        volatility=0.03,
        drift=-0.001,
        supply_disruption_day=60,
        disruption_severity=0.1,
        random_seed=42
    )
    resource_results = resource_sim.run_simulation()
    
    # Create and run product popularity simulation
    print("Running Product Popularity Simulation...")
    product_sim = SimulatorRegistry.create(
        "ProductPopularity",
        start_demand=1000,
        days=100,
        growth_rate=0.01,
        marketing_impact=0.005,
        promotion_day=70,
        promotion_effectiveness=0.2,
        random_seed=42
    )
    product_results = product_sim.run_simulation()
    
    # Create and run queueing system simulation
    print("Running Queueing System Simulation...")
    queue_sim = SimulatorRegistry.create(
        "QueueingSystem",
        max_time=100,
        arrival_rate=1.0,  # 1 customer per time unit
        service_rate=1.2,  # Can serve 1.2 customers per time unit
        num_servers=1,
        time_step=1.0,
        random_seed=42
    )
    queue_results = queue_sim.run_simulation()
    queue_stats = queue_sim.get_statistics()
    print(f"Queueing System Statistics: {queue_stats}")
    
    # Plot all simulation results
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(stock_results)
    plt.title("Stock Market Simulation")
    plt.xlabel("Day")
    plt.ylabel("Stock Price")
    
    plt.subplot(2, 2, 2)
    plt.plot(resource_results)
    plt.title("Resource Fluctuations Simulation")
    plt.xlabel("Day")
    plt.ylabel("Resource Price")
    
    plt.subplot(2, 2, 3)
    plt.plot(product_results)
    plt.title("Product Popularity Simulation")
    plt.xlabel("Day")
    plt.ylabel("Demand")
    
    plt.subplot(2, 2, 4)
    plt.plot(queue_results)
    plt.title("Queueing System Simulation")
    plt.xlabel("Time")
    plt.ylabel("Queue Length")
    
    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Saved simulation results to 'simulation_results.png'")
    
    # Define and register a custom simulator
    define_and_register_custom_simulator()


# Define a custom simulation type
class CustomSimulation(BaseSimulation):
    """A custom simulation that simulates a simple oscillator with random noise."""
    
    def __init__(
        self, days: int, amplitude: float, frequency: float, 
        noise_level: float = 0.1, random_seed: Optional[int] = None
    ) -> None:
        """Initialize the custom simulation.
        
        Args:
            days: The duration of the simulation in days.
            amplitude: The amplitude of the oscillation.
            frequency: The frequency of the oscillation (cycles per day).
            noise_level: The level of random noise to add.
            random_seed: Seed for random number generation.
        """
        super().__init__(days=days, random_seed=random_seed)
        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_level = noise_level
    
    def run_simulation(self) -> List[float]:
        """Run the simulation and return results.
        
        Returns:
            A list of values representing the oscillation with noise.
        """
        self.reset()
        
        # Generate time points
        t = np.arange(0, self.days, 1)
        
        # Generate oscillation
        base_signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t / self.days)
        
        # Add random noise
        noise = np.random.normal(0, self.noise_level, len(t))
        signal = base_signal + noise
        
        return signal.tolist()
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation."""
        params = super().get_parameters_info()
        
        params.update({
            'amplitude': {
                'type': 'float',
                'description': 'The amplitude of the oscillation',
                'required': True
            },
            'frequency': {
                'type': 'float',
                'description': 'The frequency of the oscillation (cycles per day)',
                'required': True
            },
            'noise_level': {
                'type': 'float',
                'description': 'The level of random noise to add',
                'required': False,
                'default': 0.1
            }
        })
        
        return params


def define_and_register_custom_simulator():
    """Define and register a custom simulator, then run it."""
    # Register our custom simulator
    SimulatorRegistry.register("OscillatorModel")(CustomSimulation)
    
    print("\nRegistered custom simulator 'OscillatorModel'")
    print("Updated available simulators:")
    for sim_name in SimulatorRegistry.list_simulators():
        print(f"  - {sim_name}")
    
    # Create and run the custom simulator
    print("\nRunning Custom Oscillator Simulation...")
    custom_sim = SimulatorRegistry.create(
        "OscillatorModel",
        days=100,
        amplitude=10.0,
        frequency=2.0,  # 2 full cycles
        noise_level=0.5,
        random_seed=42
    )
    custom_results = custom_sim.run_simulation()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(custom_results)
    plt.title("Custom Oscillator Simulation")
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig("custom_simulation.png")
    print("Saved custom simulation results to 'custom_simulation.png'")


if __name__ == "__main__":
    run_example()