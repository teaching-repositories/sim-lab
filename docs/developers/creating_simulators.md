# Creating New Simulators

This guide explains how to create new simulation types for SimLab, ensuring they integrate properly with the framework.

## Overview

Creating a new simulator in SimLab involves:

1. Creating a new class that inherits from `BaseSimulation`
2. Implementing the required methods
3. Registering the simulator with the registry
4. Adding appropriate documentation

## Basic Simulator Structure

Here's the basic structure of a simulator:

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
from typing import List, Union, Optional, Dict, Any

@SimulatorRegistry.register("MySimulator")
class MyCustomSimulation(BaseSimulation):
    """My custom simulation implementation.
    
    This simulation models [describe what it models].
    
    Attributes:
        param1: Description of param1
        param2: Description of param2
    """
    
    def __init__(
        self,
        days: int,
        param1: float,
        param2: str,
        random_seed: Optional[int] = None
    ):
        """Initialize the simulation.
        
        Args:
            days: Duration of the simulation in days/steps
            param1: Description of param1
            param2: Description of param2
            random_seed: Seed for random number generation
        """
        super().__init__(days=days, random_seed=random_seed)
        
        # Store parameters
        self.param1 = param1
        self.param2 = param2
        
        # Initialize simulation state
        self.results = []
    
    def run_simulation(self) -> List[Union[float, int]]:
        """Run the simulation and return results.
        
        Returns:
            A list of simulation results over time
        """
        # Reset results
        self.results = []
        
        # Run simulation for the specified number of days
        for day in range(self.days):
            # Calculate result for this day
            result = self._calculate_day_result(day)
            
            # Store result
            self.results.append(result)
        
        return self.results
    
    def _calculate_day_result(self, day: int) -> float:
        """Calculate the result for a specific day.
        
        Args:
            day: The current day
            
        Returns:
            The result for this day
        """
        # Implement your simulation logic here
        return 0.0
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata
        """
        params = super().get_parameters_info()
        params.update({
            'param1': {
                'type': 'float',
                'description': 'Description of param1',
                'required': True
            },
            'param2': {
                'type': 'str',
                'description': 'Description of param2',
                'required': True
            }
        })
        return params
```

## Key Components

### Class Declaration and Registration

```python
@SimulatorRegistry.register("MySimulator")
class MyCustomSimulation(BaseSimulation):
```

The `@SimulatorRegistry.register()` decorator registers your simulation with the registry. You can provide a name for the registry; if omitted, the class name will be used.

### Constructor

```python
def __init__(
    self,
    days: int,
    param1: float,
    param2: str,
    random_seed: Optional[int] = None
):
    super().__init__(days=days, random_seed=random_seed)
    
    # Store parameters
    self.param1 = param1
    self.param2 = param2
    
    # Initialize simulation state
    self.results = []
```

The constructor must:
1. Call `super().__init__()` with the `days` and `random_seed` parameters
2. Store simulation-specific parameters
3. Initialize the simulation state

### run_simulation Method

```python
def run_simulation(self) -> List[Union[float, int]]:
    # Reset results
    self.results = []
    
    # Run simulation for the specified number of days
    for day in range(self.days):
        # Calculate result for this day
        result = self._calculate_day_result(day)
        
        # Store result
        self.results.append(result)
    
    return self.results
```

This method must:
1. Run the simulation for the specified number of days
2. Return the results as a list

### Parameter Information

```python
@classmethod
def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
    params = super().get_parameters_info()
    params.update({
        'param1': {
            'type': 'float',
            'description': 'Description of param1',
            'required': True
        },
        'param2': {
            'type': 'str',
            'description': 'Description of param2',
            'required': True
        }
    })
    return params
```

This method returns metadata about the simulation parameters, which is used for:
1. Documentation
2. Parameter validation
3. User interfaces

## Best Practices

### Type Annotations

Always use type annotations for parameters and return values. This:
- Improves code readability
- Enables static type checking
- Helps with documentation

### Documentation

Include comprehensive documentation:
- Class docstring explaining what the simulation models
- Parameter docstrings explaining each parameter
- Method docstrings explaining each method

### Parameter Validation

Validate parameters in the constructor:

```python
def __init__(self, param1: float, ...):
    if param1 <= 0:
        raise ValueError("param1 must be positive")
```

### Encapsulation

Use private methods (prefixed with `_`) for internal implementation details:

```python
def _calculate_day_result(self, day: int) -> float:
    # Implementation details
```

### Random Number Generation

Always use the random generators initialized by the base class:

```python
# Use these:
import random
import numpy as np

value1 = random.random()
value2 = np.random.randn()
```

This ensures reproducibility when a random seed is provided.

## Example: Complete Simulator

Here's a complete example of a simple simulator:

```python
from sim_lab.core import BaseSimulation, SimulatorRegistry
import numpy as np
from typing import List, Union, Optional, Dict, Any

@SimulatorRegistry.register("RandomWalk")
class RandomWalkSimulation(BaseSimulation):
    """Simple random walk simulation.
    
    This simulation models a 1D random walk where the position changes
    by a random amount each day based on a normal distribution.
    
    Attributes:
        start_position: The starting position for the random walk
        step_size: The standard deviation of the normal distribution used for steps
    """
    
    def __init__(
        self,
        days: int,
        start_position: float = 0.0,
        step_size: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """Initialize the random walk simulation.
        
        Args:
            days: Duration of the simulation in days
            start_position: The starting position for the random walk
            step_size: The standard deviation of the normal distribution used for steps
            random_seed: Seed for random number generation
        """
        super().__init__(days=days, random_seed=random_seed)
        
        # Validate parameters
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        
        # Store parameters
        self.start_position = start_position
        self.step_size = step_size
        
        # Initialize simulation state
        self.positions = []
    
    def run_simulation(self) -> List[float]:
        """Run the simulation and return results.
        
        Returns:
            A list of positions over time
        """
        # Reset positions
        self.positions = [self.start_position]
        
        # Run simulation for the specified number of days
        for day in range(1, self.days):
            # Get previous position
            prev_position = self.positions[-1]
            
            # Calculate new position
            step = np.random.normal(0, self.step_size)
            new_position = prev_position + step
            
            # Store new position
            self.positions.append(new_position)
        
        return self.positions
    
    @classmethod
    def get_parameters_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters required by this simulation.
        
        Returns:
            A dictionary mapping parameter names to their metadata
        """
        params = super().get_parameters_info()
        params.update({
            'start_position': {
                'type': 'float',
                'description': 'The starting position for the random walk',
                'required': False,
                'default': 0.0
            },
            'step_size': {
                'type': 'float',
                'description': 'The standard deviation of the normal distribution used for steps',
                'required': False,
                'default': 1.0
            }
        })
        return params
```

## Adding Visualization

You can add visualization methods to your simulator:

```python
def plot_results(self, figsize=(10, 6)):
    """Plot the simulation results.
    
    Args:
        figsize: Figure size as a tuple of (width, height)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    plt.plot(self.positions)
    plt.title('Random Walk Simulation')
    plt.xlabel('Day')
    plt.ylabel('Position')
    plt.grid(True)
    plt.show()
```

## Testing Your Simulator

Create tests for your simulator to ensure it works as expected:

```python
import pytest
from sim_lab.core import SimulatorRegistry

def test_random_walk():
    # Create the simulator
    sim = SimulatorRegistry.create(
        "RandomWalk", 
        days=100, 
        start_position=10.0,
        step_size=2.0,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Check results
    assert len(results) == 100
    assert results[0] == 10.0
    
    # Check that results are deterministic with a seed
    sim2 = SimulatorRegistry.create(
        "RandomWalk", 
        days=100, 
        start_position=10.0,
        step_size=2.0,
        random_seed=42
    )
    results2 = sim2.run_simulation()
    assert results == results2
```

## Documentation Requirements

For each new simulator, provide:

1. A Markdown file in the `docs/simulations/` directory
2. Example usage code snippets
3. Explanation of the model and its parameters
4. Visualization examples

## Conclusion

By following these guidelines, you can create robust, well-documented simulators that integrate seamlessly with the SimLab framework. The framework handles much of the boilerplate, allowing you to focus on the simulation logic specific to your model.