# SimLab Architecture

This document outlines the high-level architecture of the SimLab framework, explaining its components, design principles, and how they interact.

## Overview

SimLab is designed around a modular, extensible architecture that allows for:

1. Consistent interface across different simulation types
2. Dynamic discovery and instantiation of simulation models
3. Easy extension with new simulation types
4. Clear separation of concerns between components

The architecture follows object-oriented principles, with a focus on:

- **Inheritance**: All simulators inherit from a common base class
- **Abstraction**: Complex simulation logic is encapsulated in specific classes
- **Polymorphism**: Common interface across different simulator types
- **Encapsulation**: Internal simulation state is managed by each simulator

## Core Components

### Base Simulation

At the heart of SimLab is the `BaseSimulation` abstract class, which defines the common interface for all simulations:

```
BaseSimulation
├── run_simulation()
├── reset()
├── get_parameters_info()
└── _initialize_random_generators()
```

This class ensures all simulators provide a consistent interface and handles common functionality like random number generation.

### Registry System

The `SimulatorRegistry` provides dynamic discovery and instantiation of simulation models:

```
SimulatorRegistry
├── register()
├── unregister()
├── get()
├── list_simulators()
├── create()
└── load_simulator_from_path()
```

The registry allows simulations to be referenced by name, decoupling the code that uses simulations from the specific simulation implementations.

### Simulation Categories

SimLab organizes simulations into several categories, each with its own specialized base class:

```
BaseSimulation
├── BasicSimulation
│   ├── StockMarketSimulation
│   ├── ResourceFluctuationsSimulation
│   └── ProductPopularitySimulation
├── DiscreteEventSimulation
│   └── QueueingSimulation
├── StatisticalSimulation
│   ├── MonteCarloSimulation
│   └── MarkovChainSimulation
├── AgentBasedSimulation
├── SystemDynamicsSimulation
├── NetworkSimulation
├── EcologicalSimulation
│   └── PredatorPreySimulation
└── DomainSpecificSimulation
    ├── EpidemiologicalSimulation
    ├── CellularAutomatonSimulation
    └── SupplyChainSimulation
```

## Directory Structure

The project follows a clear directory structure:

```
sim_lab/
├── __init__.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── base_simulation.py
│   ├── registry.py
│   └── [specific simulation files]
├── tui/
│   ├── __init__.py
│   └── app.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   └── validation.py
├── viz/
│   ├── __init__.py
│   └── plots.py
└── web/
    ├── __init__.py
    └── app.py
```

- **core/**: Contains the simulation models and core functionality
- **cli/**: Command-line interface
- **tui/**: Text-based user interface
- **web/**: Web-based interface
- **config/**: Configuration settings
- **utils/**: Utility functions
- **viz/**: Visualization components

## Interfaces

SimLab provides multiple interfaces for interacting with simulations:

1. **Python API**: Direct use in Python code
2. **CLI**: Command-line interface
3. **TUI**: Text-based user interface
4. **Web**: Web-based interface

Each interface uses the same underlying simulation models, providing different ways to access the functionality.

## Design Patterns

SimLab uses several design patterns:

1. **Factory Pattern**: The registry acts as a factory for creating simulations
2. **Singleton Pattern**: The registry is a singleton
3. **Decorator Pattern**: Registration is done through decorators
4. **Strategy Pattern**: Different simulation algorithms implement the same interface
5. **Observer Pattern**: Some simulations use observers to track state changes

## Extension Points

SimLab can be extended in several ways:

1. **New Simulation Types**: Create a new class inheriting from `BaseSimulation`
2. **Custom Visualizations**: Add new visualization methods
3. **Additional Interfaces**: Create new interfaces to the simulation models
4. **Plugins**: Develop plugins that can be loaded at runtime

## Data Flow

A typical data flow through the system:

1. User selects a simulation type and parameters
2. Interface code creates a simulation instance through the registry
3. Simulation runs and generates results
4. Results are processed, visualized, or exported

## Dependencies

SimLab has minimal external dependencies:

- **NumPy**: For numerical operations
- **Matplotlib** (optional): For visualization
- **Rich** (optional): For TUI
- **Flask** (optional): For web interface

## Future Architecture Directions

Planned architectural improvements:

1. **Plugin System**: More formal plugin architecture
2. **Distributed Simulations**: Support for distributed computing
3. **Cloud Integration**: Deployment to cloud environments
4. **Real-time Visualization**: Live updating visualizations
5. **Interoperability**: Better integration with other tools