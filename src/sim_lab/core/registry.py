"""Registry for simulation models."""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar

from .base_simulation import BaseSimulation

T = TypeVar('T', bound=BaseSimulation)


class SimulatorRegistry:
    """Registry for simulation models.
    
    This class maintains a registry of all available simulation models,
    allowing for dynamic discovery and instantiation of simulations.
    """
    
    _registry: Dict[str, Type[BaseSimulation]] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> callable:
        """Decorator to register a simulation class.
        
        Args:
            name: The name to register the simulation under. If None, 
                 the class name will be used.
                 
        Returns:
            A decorator function that registers the class.
        """
        def decorator(sim_class: Type[T]) -> Type[T]:
            if not inspect.isclass(sim_class) or not issubclass(sim_class, BaseSimulation):
                raise TypeError(f"Class {sim_class.__name__} must be a subclass of BaseSimulation")
            
            sim_name = name if name is not None else sim_class.__name__
            cls._registry[sim_name] = sim_class
            return sim_class
        
        return decorator
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a simulation from the registry.
        
        Args:
            name: The name of the simulation to remove.
            
        Raises:
            KeyError: If the simulation is not registered.
        """
        if name in cls._registry:
            del cls._registry[name]
        else:
            raise KeyError(f"Simulation '{name}' is not registered")
    
    @classmethod
    def get(cls, name: str) -> Type[BaseSimulation]:
        """Get a simulation class by name.
        
        Args:
            name: The name of the simulation to get.
            
        Returns:
            The simulation class.
            
        Raises:
            KeyError: If the simulation is not registered.
        """
        if name in cls._registry:
            return cls._registry[name]
        else:
            raise KeyError(f"Simulation '{name}' is not registered")
    
    @classmethod
    def list_simulators(cls) -> List[str]:
        """List all registered simulations.
        
        Returns:
            A list of simulation names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseSimulation:
        """Create an instance of a simulation.
        
        Args:
            name: The name of the simulation to create.
            **kwargs: Parameters to pass to the simulation constructor.
            
        Returns:
            A new instance of the requested simulation.
            
        Raises:
            KeyError: If the simulation is not registered.
        """
        sim_class = cls.get(name)
        return sim_class(**kwargs)
    
    @classmethod
    def load_simulator_from_path(cls, module_path: str, class_name: str, 
                                register_as: Optional[str] = None) -> Type[BaseSimulation]:
        """Load a simulator class from a module path and register it.
        
        Args:
            module_path: The dotted path to the module (e.g. 'sim_lab.custom.my_simulation').
            class_name: The name of the class to load.
            register_as: The name to register the simulation under. If None,
                         the class name will be used.
                         
        Returns:
            The loaded simulation class.
            
        Raises:
            ImportError: If the module or class cannot be loaded.
            TypeError: If the class is not a subclass of BaseSimulation.
        """
        try:
            module = importlib.import_module(module_path)
            sim_class = getattr(module, class_name)
            
            if not inspect.isclass(sim_class) or not issubclass(sim_class, BaseSimulation):
                raise TypeError(f"Class {class_name} must be a subclass of BaseSimulation")
            
            reg_name = register_as if register_as is not None else class_name
            cls._registry[reg_name] = sim_class
            return sim_class
            
        except ImportError:
            raise ImportError(f"Could not import module '{module_path}'")
        except AttributeError:
            raise ImportError(f"Module '{module_path}' has no class named '{class_name}'")