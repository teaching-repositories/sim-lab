# Simulator Registry System

The SimLab registry system provides a centralized mechanism for discovering, instantiating, and managing simulation models. This guide explains how the registry works and how to use it effectively.

## Overview

The registry system solves several important problems:

1. **Dynamic Discovery**: Users can discover available simulation models at runtime
2. **Loose Coupling**: Simulators are decoupled from the code that uses them
3. **Plugin Architecture**: Third-party simulators can be registered without modifying core code
4. **Configuration-Based**: Simulators can be specified by name in configuration files
5. **Centralized Management**: One place to manage available simulators

## Registry Architecture

The registry system is implemented in the `SimulatorRegistry` class, which is a singleton that maintains a mapping from simulator names to simulator classes. The core components are:

- **Registry Mapping**: A dictionary that maps simulator names to simulator classes
- **Registration Mechanism**: A decorator for registering simulator classes
- **Discovery API**: Methods for listing and retrieving registered simulators
- **Factory Method**: A method for creating instances of simulators

## Using the Registry

### Registering a Simulator

To register a simulator, use the `@SimulatorRegistry.register()` decorator:

```python
from sim_lab.core import SimulatorRegistry, BaseSimulation

@SimulatorRegistry.register("MySimulator")
class MyCustomSimulation(BaseSimulation):
    # Your simulator implementation
    pass
```

You can also register a simulator with a different name than its class name:

```python
@SimulatorRegistry.register("ShortName")
class VeryLongAndDescriptiveSimulationClassName(BaseSimulation):
    # Your simulator implementation
    pass
```

### Listing Available Simulators

To list all available simulators:

```python
from sim_lab.core import SimulatorRegistry

simulators = SimulatorRegistry.list_simulators()
print(f"Available simulators: {simulators}")
```

### Creating a Simulator Instance

To create an instance of a simulator:

```python
from sim_lab.core import SimulatorRegistry

# Create the simulator by name
simulator = SimulatorRegistry.create(
    "MySimulator",
    param1=value1,
    param2=value2
)

# Run the simulation
results = simulator.run_simulation()
```

### Loading Simulators Dynamically

You can also load simulator classes from external modules:

```python
from sim_lab.core import SimulatorRegistry

# Load a simulator from a module path
SimulatorRegistry.load_simulator_from_path(
    module_path="my_package.my_module",
    class_name="MySimulatorClass",
    register_as="MyCustomSimulator"
)

# Now you can create instances of it
simulator = SimulatorRegistry.create("MyCustomSimulator", ...)
```

## Implementation Details

### The SimulatorRegistry Class

The `SimulatorRegistry` class is a simple class with class methods that operate on a shared registry dictionary:

```python
class SimulatorRegistry:
    """Registry for simulation models."""
    
    _registry: Dict[str, Type[BaseSimulation]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> callable:
        """Decorator to register a simulation class."""
        def decorator(sim_class: Type[T]) -> Type[T]:
            if not inspect.isclass(sim_class) or not issubclass(sim_class, BaseSimulation):
                raise TypeError(f"Class {sim_class.__name__} must be a subclass of BaseSimulation")
            
            sim_name = name if name is not None else sim_class.__name__
            cls._registry[sim_name] = sim_class
            return sim_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseSimulation]:
        """Get a simulation class by name."""
        if name in cls._registry:
            return cls._registry[name]
        else:
            raise KeyError(f"Simulation '{name}' is not registered")
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseSimulation:
        """Create an instance of a simulation."""
        sim_class = cls.get(name)
        return sim_class(**kwargs)
    
    # ... other methods ...
```

### Registration Process

When a simulator class is decorated with `@SimulatorRegistry.register()`, the following happens:

1. The decorator function is called with the optional name parameter
2. The decorator function returns a decorator function that takes a class
3. When the class is defined, the decorator function is called with the class as an argument
4. The decorator function checks if the class is a subclass of `BaseSimulation`
5. If valid, the class is added to the registry under the specified name (or its class name)
6. The original class is returned, so it can be used normally

### Class Hierarchy and BaseSimulation

All simulator classes must inherit from `BaseSimulation`, which provides the common interface for all simulators:

```python
class BaseSimulation(ABC):
    """Base class for all SimLab simulations."""
    
    def __init__(self, days: int, random_seed: Optional[int] = None, **kwargs):
        """Initialize the base simulation."""
        self.days = days
        self.random_seed = random_seed
        self._initialize_random_generators()
    
    @abstractmethod
    def run_simulation(self) -> List[Union[float, int]]:
        """Run the simulation and return results."""
        pass
    
    # ... other methods ...
```

The `BaseSimulation` class defines the common interface that all simulators must implement, ensuring consistency across different simulator types.

## Best Practices

### Naming Conventions

- Use clear, descriptive names for your simulators
- Follow PascalCase for class names and CamelCase for registry names
- Use domain-specific names that reflect the simulation's purpose

### Parameter Documentation

- Document your simulator's parameters thoroughly using docstrings
- Implement the `get_parameters_info()` method to provide parameter metadata
- Include information about parameter types, descriptions, and defaults

### Error Handling

- Handle invalid parameters gracefully with clear error messages
- Validate all inputs in the constructor
- Provide helpful troubleshooting information in error messages

### Testing Registered Simulators

- Test that your simulator is properly registered
- Test that your simulator can be created through the registry
- Test that your simulator works correctly when created through the registry

## Advanced Usage

### Factory Patterns

You can create factory functions that use the registry to create simulators based on configuration:

```python
def create_simulator_from_config(config: Dict[str, Any]) -> BaseSimulation:
    """Create a simulator from a configuration dictionary."""
    simulator_type = config.pop("type")
    return SimulatorRegistry.create(simulator_type, **config)
```

### Registry Hooks

You can add hooks to the registration process to perform additional tasks when simulators are registered:

```python
# Add a hook to the register method
original_register = SimulatorRegistry.register

def register_with_hook(name=None):
    decorator = original_register(name)
    
    def wrapper(cls):
        result = decorator(cls)
        # Perform additional tasks here
        print(f"Registered simulator: {cls.__name__}")
        return result
    
    return wrapper

SimulatorRegistry.register = register_with_hook
```

### Plugin System

You can create a plugin system that automatically discovers and registers simulators from specified directories:

```python
import importlib
import pkgutil
import inspect
from sim_lab.core import BaseSimulation, SimulatorRegistry

def discover_plugins(package_name: str) -> None:
    """Discover and register all simulator plugins in a package."""
    package = importlib.import_module(package_name)
    
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_name)
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and issubclass(obj, BaseSimulation) and 
                obj.__module__ == module_name and obj != BaseSimulation):
                # Register the simulator with its class name
                SimulatorRegistry.register()(obj)
```

## Troubleshooting

### Common Issues

1. **Simulator not found**: Ensure the simulator is registered and imported
2. **Parameter errors**: Check that you're providing all required parameters
3. **Import errors**: Ensure the module path is correct when loading external simulators

### Registry Inspection

You can inspect the registry to troubleshoot issues:

```python
from sim_lab.core import SimulatorRegistry
import pprint

# Print all registered simulators
pprint.pprint(SimulatorRegistry._registry)
```

## Conclusion

The simulator registry system provides a flexible and powerful way to manage simulation models in SimLab. By using the registry, you can create modular, extensible applications that can dynamically discover and use simulation models.