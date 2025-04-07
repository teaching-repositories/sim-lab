# Agent-Based Simulation

Agent-Based Modeling (ABM) is a powerful approach for simulating complex systems through the actions and interactions of autonomous agents. The AgentBasedSimulation in SimLab provides a framework for creating, running, and analyzing agent-based models.

## Overview

The AgentBasedSimulation class allows you to model systems where individual agents follow simple rules, yet their interactions lead to emergent behavior at the system level. Key features include:

- Customizable agent behavior
- Spatial positioning and neighborhood detection
- Environment that agents can interact with
- Metrics calculation and history tracking
- Support for various agent types and interaction patterns

## Basic Usage

Here's a simple example of creating and using an agent-based simulation:

```python
from sim_lab.core import AgentBasedSimulation, Agent, Environment
import numpy as np

# Define a custom agent class
class MyAgent(Agent):
    def __init__(self, agent_id, position=None):
        initial_state = {
            "energy": 100,
            "alive": True
        }
        super().__init__(agent_id, initial_state, position)
    
    def update(self, environment, neighbors):
        # Agent loses energy each step
        self.state["energy"] -= 1
        
        # Agent dies if energy reaches 0
        if self.state["energy"] <= 0:
            self.state["alive"] = False
        
        # Agent moves randomly
        if self.position is not None:
            x, y = self.position
            x += np.random.uniform(-1, 1)
            y += np.random.uniform(-1, 1)
            self.position = (x, y)

# Define a custom environment
class MyEnvironment(Environment):
    def __init__(self):
        initial_state = {
            "resource_level": 1000,
            "temperature": 20
        }
        super().__init__(initial_state)
    
    def update(self, agents):
        # Resources are consumed by agents
        alive_agents = sum(1 for agent in agents if agent.state.get("alive", False))
        consumption = alive_agents * 0.1
        self.state["resource_level"] = max(0, self.state["resource_level"] - consumption)

# Create an agent factory function
def agent_factory(agent_id):
    return MyAgent(
        agent_id=agent_id,
        position=(np.random.random() * 100, np.random.random() * 100)
    )

# Create the environment
env = MyEnvironment()

# Create the simulation
sim = AgentBasedSimulation(
    agent_factory=agent_factory,
    num_agents=50,
    environment=env,
    days=100,
    neighborhood_radius=10.0,
    save_history=True
)

# Run the simulation
metrics = sim.run_simulation()

# Analyze results
alive_count = [m.get("alive_True", 0) for m in metrics]
print(f"Agents alive at the end: {alive_count[-1]}")
```

## Creating Agents

Agents in SimLab are created by extending the `Agent` base class:

```python
class Agent:
    def __init__(self, agent_id, initial_state=None, position=None):
        self.agent_id = agent_id
        self.state = initial_state or {}
        self.position = position
        self.history = []
    
    def update(self, environment, neighbors):
        # This method should be overridden to define agent behavior
        pass
    
    def move(self, new_position):
        self.position = new_position
    
    def get_state(self):
        return self.state
    
    def save_history(self):
        self.history.append(self.state.copy())
    
    def reset(self):
        self.history = []
```

Your custom agents should override the `update` method to define the agent's behavior at each time step. The agent can:

- Access and modify its own state
- Interact with the environment
- Interact with neighboring agents
- Move to a new position

## Creating an Environment

The environment provides a context for the agents to operate in and can change over time:

```python
class Environment:
    def __init__(self, initial_state=None, bounds=(0, 0, 100, 100)):
        self.state = initial_state or {}
        self.bounds = bounds
        self.history = []
    
    def update(self, agents):
        # This method should be overridden to define environment dynamics
        pass
    
    def get_state(self):
        return self.state
    
    def save_history(self):
        self.history.append(self.state.copy())
    
    def reset(self):
        self.history = []
```

Override the `update` method to define how the environment changes based on the agents' states and actions.

## Simulation Parameters

When creating an AgentBasedSimulation, you can specify the following parameters:

- **agent_factory**: Function that creates new agents with given IDs
- **num_agents**: Number of agents to create
- **environment**: The environment in which agents operate
- **days**: Number of steps to simulate
- **neighborhood_radius**: Radius for determining agent neighbors
- **save_history**: Whether to save agent and environment history
- **random_seed**: Seed for random number generation

## Analyzing Results

The AgentBasedSimulation class provides several methods for analyzing simulation results:

```python
# Get the history of a specific agent
agent_history = sim.get_agent_history(agent_id=0)

# Get the environment history
env_history = sim.get_environment_history()

# Get the history of a specific metric
metric_history = sim.get_metric_history(metric_name="alive_True")
```

## Example: Predator-Prey Model

This example demonstrates a predator-prey ecosystem model:

```python
from sim_lab.core import AgentBasedSimulation, Agent, Environment
import numpy as np
import matplotlib.pyplot as plt

# Define prey agent
class PreyAgent(Agent):
    def __init__(self, agent_id, position=None):
        initial_state = {
            "type": "prey",
            "energy": 20,
            "age": 0,
            "reproduce_countdown": np.random.randint(5, 10)
        }
        super().__init__(agent_id, initial_state, position)
    
    def update(self, environment, neighbors):
        # Age the prey
        self.state["age"] += 1
        
        # Lose energy
        self.state["energy"] -= 1
        
        # Reproduce if ready
        self.state["reproduce_countdown"] -= 1
        
        # Move away from predators
        predators = [n for n in neighbors if n.state["type"] == "predator"]
        if predators:
            # Move away from the closest predator
            closest = predators[0]
            dx = self.position[0] - closest.position[0]
            dy = self.position[1] - closest.position[1]
            dist = max(0.1, (dx**2 + dy**2)**0.5)
            
            x, y = self.position
            x = max(0, min(100, x + dx/dist * 3))
            y = max(0, min(100, y + dy/dist * 3))
            self.position = (x, y)
        else:
            # Random movement
            x, y = self.position
            x = max(0, min(100, x + np.random.uniform(-5, 5)))
            y = max(0, min(100, y + np.random.uniform(-5, 5)))
            self.position = (x, y)
        
        # Find food in environment
        if environment.state.get("grass", 0) > 0:
            self.state["energy"] += 5
            environment.state["grass"] -= 1

# Define predator agent
class PredatorAgent(Agent):
    def __init__(self, agent_id, position=None):
        initial_state = {
            "type": "predator",
            "energy": 30,
            "age": 0,
            "reproduce_countdown": np.random.randint(10, 15)
        }
        super().__init__(agent_id, initial_state, position)
    
    def update(self, environment, neighbors):
        # Age the predator
        self.state["age"] += 1
        
        # Lose energy
        self.state["energy"] -= 2
        
        # Reproduce if ready
        self.state["reproduce_countdown"] -= 1
        
        # Chase prey
        prey = [n for n in neighbors if n.state["type"] == "prey"]
        if prey and self.state["energy"] < 50:
            # Move toward the closest prey
            closest = prey[0]
            dx = closest.position[0] - self.position[0]
            dy = closest.position[1] - self.position[1]
            dist = max(0.1, (dx**2 + dy**2)**0.5)
            
            x, y = self.position
            x = max(0, min(100, x + dx/dist * 5))
            y = max(0, min(100, y + dy/dist * 5))
            self.position = (x, y)
            
            # If close enough, eat prey
            if dist < 5:
                self.state["energy"] += 15
                environment.state["prey_eaten"] = environment.state.get("prey_eaten", 0) + 1
                closest.state["energy"] = 0  # Mark for removal
        else:
            # Random movement
            x, y = self.position
            x = max(0, min(100, x + np.random.uniform(-3, 3)))
            y = max(0, min(100, y + np.random.uniform(-3, 3)))
            self.position = (x, y)

# Define environment
class EcosystemEnvironment(Environment):
    def __init__(self):
        initial_state = {
            "grass": 100,
            "prey_eaten": 0,
            "day": 0
        }
        super().__init__(initial_state)
    
    def update(self, agents):
        # Grow more grass
        self.state["grass"] = min(200, self.state["grass"] + 5)
        self.state["day"] += 1

# Create agent factory function
def agent_factory(agent_id):
    if agent_id % 4 == 0:
        return PredatorAgent(agent_id)
    else:
        return PreyAgent(agent_id)

# Create simulation
sim = AgentBasedSimulation(
    agent_factory=agent_factory,
    num_agents=100,
    environment=EcosystemEnvironment(),
    days=100,
    neighborhood_radius=10.0,
    save_history=True,
    random_seed=42
)

# Run simulation
metrics = sim.run_simulation()

# Extract predator and prey counts
predator_counts = [m.get("type_predator", 0) for m in metrics]
prey_counts = [m.get("type_prey", 0) for m in metrics]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(predator_counts, label='Predators')
plt.plot(prey_counts, label='Prey')
plt.title('Predator-Prey Population Dynamics')
plt.xlabel('Time Step')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
```

## Advanced Topics

### Agent Communication

Agents can communicate with each other through their state attributes:

```python
def update(self, environment, neighbors):
    # Send a message to neighbors
    self.state["message"] = "Hello, neighbors!"
    
    # Read messages from neighbors
    for neighbor in neighbors:
        message = neighbor.state.get("message")
        if message:
            # Process the message
            pass
```

### Agent Death and Birth

Agents can "die" by setting a state flag that you check in metrics calculation:

```python
def update(self, environment, neighbors):
    if self.state["energy"] <= 0:
        self.state["alive"] = False
```

To add new agents during the simulation, you would need to modify the AgentBasedSimulation class to allow for agent creation and removal.

### Heterogeneous Agents

You can create different types of agents in your agent factory function:

```python
def agent_factory(agent_id):
    agent_type = agent_id % 3
    
    if agent_type == 0:
        return ProducerAgent(agent_id)
    elif agent_type == 1:
        return ConsumerAgent(agent_id)
    else:
        return RecyclerAgent(agent_id)
```

### Spatial Patterns

Agents with position attributes can form spatial patterns:

```python
def update(self, environment, neighbors):
    # Flocking behavior
    if neighbors:
        # Calculate average position of neighbors
        avg_x = sum(n.position[0] for n in neighbors) / len(neighbors)
        avg_y = sum(n.position[1] for n in neighbors) / len(neighbors)
        
        # Move toward average position
        x, y = self.position
        x += (avg_x - x) * 0.1
        y += (avg_y - y) * 0.1
        self.position = (x, y)
```

## API Reference

### AgentBasedSimulation Class

```python
AgentBasedSimulation(
    agent_factory: Callable[[int], Agent],
    num_agents: int,
    environment: Optional[Environment] = None,
    days: int = 100,
    neighborhood_radius: float = 10.0,
    save_history: bool = False,
    random_seed: Optional[int] = None
)
```

#### Methods

- **get_agent_neighbors(agent)**: Get the neighbors of an agent based on proximity
- **calculate_metrics()**: Calculate metrics for the current simulation state
- **run_simulation()**: Run the agent-based simulation
- **get_agent_history(agent_id)**: Get the state history for a specific agent
- **get_environment_history()**: Get the environment state history
- **get_metric_history(metric_name)**: Get the history of a specific metric

### Agent Class

```python
Agent(
    agent_id: int,
    initial_state: Dict[str, Any] = None,
    position: Optional[Tuple[float, float]] = None
)
```

#### Methods

- **update(environment, neighbors)**: Update the agent's state
- **move(new_position)**: Move the agent to a new position
- **get_state()**: Get the current state of the agent
- **save_history()**: Save the current state to the agent's history
- **reset()**: Reset the agent's history

### Environment Class

```python
Environment(
    initial_state: Dict[str, Any] = None,
    bounds: Tuple[float, float, float, float] = (0, 0, 100, 100)
)
```

#### Methods

- **update(agents)**: Update the environment based on agent states
- **get_state()**: Get the current state of the environment
- **save_history()**: Save the current state to the environment's history
- **reset()**: Reset the environment's history

## Further Reading

For more information about agent-based modeling, see:

- [An Introduction to Agent-Based Modeling](https://mitpress.mit.edu/books/introduction-agent-based-modeling) by Uri Wilensky and William Rand
- [Agent-Based and Individual-Based Modeling: A Practical Introduction](https://press.princeton.edu/books/paperback/9780691190839/agent-based-and-individual-based-modeling) by Steven F. Railsback and Volker Grimm
- [NetLogo](http://ccl.northwestern.edu/netlogo/) - A popular platform for agent-based modeling
- [Mesa](https://mesa.readthedocs.io/) - A Python framework for agent-based modeling