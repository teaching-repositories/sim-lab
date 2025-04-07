"""
Example of using SimLab to create a predator-prey simulation using the agent-based framework.

This example demonstrates:
1. Defining custom Agent subclasses for predators and prey
2. Creating an environment with food resources
3. Implementing agent behaviors and interactions
4. Visualizing the spatial distribution and population dynamics
"""

from sim_lab.core import SimulatorRegistry, Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Tuple, Dict, Any


class AnimalAgent(Agent):
    """Base class for animal agents in the ecosystem."""
    
    def __init__(self, agent_id: int, position: Tuple[float, float], energy: float = 100):
        """Initialize a new animal agent.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position (x, y)
            energy: Initial energy level
        """
        initial_state = {
            "energy": energy,
            "age": 0,
            "alive": True
        }
        super().__init__(agent_id, initial_state, position)
    
    def move(self, environment: 'EcosystemEnvironment', neighbors: List[Agent]) -> None:
        """Move randomly within the environment."""
        if not self.state["alive"]:
            return
            
        # Random movement
        move_distance = 2.0
        angle = random.uniform(0, 2 * np.pi)
        dx = move_distance * np.cos(angle)
        dy = move_distance * np.sin(angle)
        
        # New position with boundary checking
        new_x = min(max(self.position[0] + dx, environment.bounds[0]), environment.bounds[2])
        new_y = min(max(self.position[1] + dy, environment.bounds[1]), environment.bounds[3])
        
        self.position = (new_x, new_y)
        
        # Movement consumes energy
        self.state["energy"] -= 1
        
        # Die if energy depleted
        if self.state["energy"] <= 0:
            self.state["alive"] = False
    
    def age_up(self) -> None:
        """Increase age by one time step."""
        self.state["age"] += 1
        # Die of old age if too old
        if self.state["age"] > self.max_age:
            self.state["alive"] = False


class PreyAgent(AnimalAgent):
    """Prey animal that eats plants and can be eaten by predators."""
    
    def __init__(self, agent_id: int, position: Tuple[float, float], energy: float = 100):
        super().__init__(agent_id, position, energy)
        
        # Prey-specific attributes
        self.state["type"] = "prey"
        self.state["reproduction_threshold"] = 150
        self.max_age = 100
    
    def update(self, environment: 'EcosystemEnvironment', neighbors: List[Agent]) -> None:
        """Update the prey's state based on environment and neighbors."""
        if not self.state["alive"]:
            return
            
        # Move
        self.move(environment, neighbors)
        
        # Age
        self.age_up()
        
        if not self.state["alive"]:
            return
            
        # Eat food if available nearby
        self.find_and_eat_food(environment)
        
        # Reproduce if enough energy
        if self.state["energy"] >= self.state["reproduction_threshold"]:
            self.state["ready_to_reproduce"] = True
    
    def find_and_eat_food(self, environment: 'EcosystemEnvironment') -> None:
        """Find and consume food in the environment."""
        food_found = environment.consume_food_at_location(self.position, 3.0)
        if food_found:
            self.state["energy"] += 20


class PredatorAgent(AnimalAgent):
    """Predator animal that hunts and eats prey."""
    
    def __init__(self, agent_id: int, position: Tuple[float, float], energy: float = 150):
        super().__init__(agent_id, position, energy)
        
        # Predator-specific attributes
        self.state["type"] = "predator"
        self.state["reproduction_threshold"] = 200
        self.max_age = 150
        self.hunting_distance = 5.0
    
    def update(self, environment: 'EcosystemEnvironment', neighbors: List[Agent]) -> None:
        """Update the predator's state based on environment and neighbors."""
        if not self.state["alive"]:
            return
            
        # Move
        self.move(environment, neighbors)
        
        # Age
        self.age_up()
        
        if not self.state["alive"]:
            return
            
        # Hunt prey if any are nearby
        self.hunt_prey(neighbors)
        
        # Reproduce if enough energy
        if self.state["energy"] >= self.state["reproduction_threshold"]:
            self.state["ready_to_reproduce"] = True
    
    def hunt_prey(self, neighbors: List[Agent]) -> None:
        """Hunt for prey among neighboring agents."""
        for neighbor in neighbors:
            # Check if the neighbor is a prey and alive
            if (neighbor.state.get("type") == "prey" and 
                neighbor.state.get("alive", False)):
                
                # Calculate distance to prey
                dx = self.position[0] - neighbor.position[0]
                dy = self.position[1] - neighbor.position[1]
                distance = (dx**2 + dy**2)**0.5
                
                # If prey is within hunting distance, catch and eat it
                if distance <= self.hunting_distance:
                    # Mark prey as dead
                    neighbor.state["alive"] = False
                    
                    # Gain energy from eating
                    self.state["energy"] += 50
                    return  # Only eat one prey per time step


class EcosystemEnvironment(Environment):
    """Environment representing an ecosystem with food resources."""
    
    def __init__(self, bounds: Tuple[float, float, float, float] = (0, 0, 100, 100),
                 initial_food_density: float = 0.2):
        """Initialize the ecosystem environment.
        
        Args:
            bounds: The spatial bounds (x_min, y_min, x_max, y_max)
            initial_food_density: Initial density of food in the environment
        """
        # Calculate environment dimensions
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Initialize food resources as a grid
        grid_size = 10
        grid_width = int(width / grid_size)
        grid_height = int(height / grid_size)
        
        food_grid = np.zeros((grid_width, grid_height))
        for i in range(grid_width):
            for j in range(grid_height):
                if random.random() < initial_food_density:
                    food_grid[i, j] = random.uniform(10, 30)  # Random amount of food
        
        initial_state = {
            "food_grid": food_grid,
            "grid_size": grid_size,
            "regrowth_rate": 0.05,  # Rate at which food regrows
            "prey_count": 0,
            "predator_count": 0,
            "prey_born": 0,
            "predator_born": 0,
            "prey_died": 0,
            "predator_died": 0
        }
        
        super().__init__(initial_state, bounds)
    
    def update(self, agents: List[Agent]) -> None:
        """Update the environment state based on the agents.
        
        Args:
            agents: List of all agents in the simulation
        """
        # Reset counts for this time step
        self.state["prey_count"] = 0
        self.state["predator_count"] = 0
        
        # Count living agents by type
        for agent in agents:
            if agent.state.get("alive", False):
                if agent.state.get("type") == "prey":
                    self.state["prey_count"] += 1
                elif agent.state.get("type") == "predator":
                    self.state["predator_count"] += 1
        
        # Update food resources (regrowth)
        self._regrow_food()
    
    def _regrow_food(self) -> None:
        """Regrow food in the environment."""
        food_grid = self.state["food_grid"]
        for i in range(food_grid.shape[0]):
            for j in range(food_grid.shape[1]):
                # Food grows based on the regrowth rate
                if random.random() < self.state["regrowth_rate"]:
                    food_grid[i, j] = min(food_grid[i, j] + random.uniform(1, 3), 30)
    
    def consume_food_at_location(self, position: Tuple[float, float], radius: float) -> bool:
        """Attempt to consume food at the given location.
        
        Args:
            position: Position (x, y) to check for food
            radius: Search radius for finding food
            
        Returns:
            True if food was found and consumed, False otherwise
        """
        # Convert position to grid coordinates
        grid_size = self.state["grid_size"]
        grid_x = int(position[0] / grid_size)
        grid_y = int(position[1] / grid_size)
        
        # Check bounds
        if (0 <= grid_x < self.state["food_grid"].shape[0] and
            0 <= grid_y < self.state["food_grid"].shape[1]):
            
            # If there's food at this location, consume some of it
            if self.state["food_grid"][grid_x, grid_y] > 0:
                # Consume up to 5 units of food
                amount_to_consume = min(5, self.state["food_grid"][grid_x, grid_y])
                self.state["food_grid"][grid_x, grid_y] -= amount_to_consume
                return True
        
        return False
    
    def get_food_grid(self) -> np.ndarray:
        """Get the current food grid."""
        return self.state["food_grid"]


def create_agent_factory(agent_class, environment_bounds):
    """Create a factory function for generating agents.
    
    Args:
        agent_class: The class of agent to create (PreyAgent or PredatorAgent)
        environment_bounds: The spatial bounds of the environment
        
    Returns:
        A function that creates a new agent with a given ID
    """
    def factory(agent_id):
        # Generate a random position within the environment bounds
        x = random.uniform(environment_bounds[0], environment_bounds[2])
        y = random.uniform(environment_bounds[1], environment_bounds[3])
        
        # Create a new agent
        return agent_class(agent_id, (x, y))
    
    return factory


def handle_reproduction(simulation):
    """Handle reproduction of agents."""
    new_agents = []
    next_id = len(simulation.agents)
    
    for agent in simulation.agents:
        if agent.state.get("ready_to_reproduce", False) and agent.state.get("alive", False):
            # Create a new agent of the same type
            agent_class = PreyAgent if agent.state.get("type") == "prey" else PredatorAgent
            
            # Create offspring near parent
            offspring_x = agent.position[0] + random.uniform(-5, 5)
            offspring_y = agent.position[1] + random.uniform(-5, 5)
            
            # Keep position within bounds
            bounds = simulation.environment.bounds
            offspring_x = min(max(offspring_x, bounds[0]), bounds[2])
            offspring_y = min(max(offspring_y, bounds[1]), bounds[3])
            
            # Create the new agent
            new_agent = agent_class(next_id, (offspring_x, offspring_y))
            new_agents.append(new_agent)
            next_id += 1
            
            # Parent loses energy after reproduction
            agent.state["energy"] /= 2
            agent.state["ready_to_reproduce"] = False
            
            # Update counters in environment
            if agent.state.get("type") == "prey":
                simulation.environment.state["prey_born"] += 1
            else:
                simulation.environment.state["predator_born"] += 1
    
    # Add new agents to the simulation
    simulation.agents.extend(new_agents)


def custom_metric_calculator(simulation):
    """Calculate custom metrics for the predator-prey simulation."""
    # Count living agents
    prey_alive = 0
    predator_alive = 0
    prey_energy = []
    predator_energy = []
    
    for agent in simulation.agents:
        if agent.state.get("alive", False):
            if agent.state.get("type") == "prey":
                prey_alive += 1
                prey_energy.append(agent.state.get("energy", 0))
            elif agent.state.get("type") == "predator":
                predator_alive += 1
                predator_energy.append(agent.state.get("energy", 0))
    
    # Calculate average energy if there are living agents
    avg_prey_energy = sum(prey_energy) / max(1, len(prey_energy))
    avg_predator_energy = sum(predator_energy) / max(1, len(predator_energy))
    
    # Calculate total food in environment
    total_food = np.sum(simulation.environment.state["food_grid"])
    
    # Return metrics
    return {
        "prey_count": prey_alive,
        "predator_count": predator_alive,
        "avg_prey_energy": avg_prey_energy,
        "avg_predator_energy": avg_predator_energy,
        "total_food": total_food,
        "prey_born": simulation.environment.state["prey_born"],
        "predator_born": simulation.environment.state["predator_born"],
        "prey_died": simulation.environment.state["prey_died"],
        "predator_died": simulation.environment.state["predator_died"]
    }


# Main simulation function
def run_predator_prey_simulation(
    num_prey=100,
    num_predators=20,
    days=200,
    environment_size=(0, 0, 100, 100),
    initial_food_density=0.3
):
    """Run a predator-prey agent-based simulation.
    
    Args:
        num_prey: Initial number of prey agents
        num_predators: Initial number of predator agents
        days: Number of days (time steps) to simulate
        environment_size: Bounds of the environment (x_min, y_min, x_max, y_max)
        initial_food_density: Initial density of food resources
        
    Returns:
        The completed simulation
    """
    # Create environment
    environment = EcosystemEnvironment(
        bounds=environment_size,
        initial_food_density=initial_food_density
    )
    
    # Create simulation with prey agents
    prey_factory = create_agent_factory(PreyAgent, environment_size)
    sim = SimulatorRegistry.create(
        "AgentBased",
        agent_factory=prey_factory,
        num_agents=num_prey,
        environment=environment,
        days=days,
        neighborhood_radius=10.0,
        save_history=True,
        random_seed=42
    )
    
    # Extend with predator agents
    predator_factory = create_agent_factory(PredatorAgent, environment_size)
    predator_start_id = num_prey
    for i in range(num_predators):
        predator = predator_factory(predator_start_id + i)
        sim.agents.append(predator)
    
    # Override default metric calculator
    sim.calculate_metrics = lambda: custom_metric_calculator(sim)
    
    # Run the simulation with a custom step function
    sim.metrics = []  # Reset metrics
    
    # Calculate initial metrics
    sim.metrics = [sim.calculate_metrics()]
    
    # Run simulation
    for day in range(1, days):
        # Update agents
        for agent in sim.agents:
            if agent.state.get("alive", False):
                neighbors = sim.get_agent_neighbors(agent)
                agent.update(sim.environment, neighbors)
                
                if sim.save_history:
                    agent.save_history()
        
        # Handle reproduction
        handle_reproduction(sim)
        
        # Update environment
        sim.environment.update(sim.agents)
        
        if sim.save_history:
            sim.environment.save_history()
        
        # Calculate metrics
        sim.metrics.append(sim.calculate_metrics())
    
    return sim


def visualize_simulation_results(sim):
    """Visualize the results of a predator-prey simulation.
    
    Args:
        sim: The completed simulation
    """
    # Extract metrics
    days = range(len(sim.metrics))
    prey_counts = [m["prey_count"] for m in sim.metrics]
    predator_counts = [m["predator_count"] for m in sim.metrics]
    
    # Plot population dynamics
    plt.figure(figsize=(12, 6))
    plt.plot(days, prey_counts, 'g-', label='Prey')
    plt.plot(days, predator_counts, 'r-', label='Predators')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('Predator-Prey Population Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot average energy levels
    plt.figure(figsize=(12, 6))
    avg_prey_energy = [m.get("avg_prey_energy", 0) for m in sim.metrics]
    avg_predator_energy = [m.get("avg_predator_energy", 0) for m in sim.metrics]
    plt.plot(days, avg_prey_energy, 'g-', label='Avg Prey Energy')
    plt.plot(days, avg_predator_energy, 'r-', label='Avg Predator Energy')
    plt.xlabel('Days')
    plt.ylabel('Energy Level')
    plt.title('Average Agent Energy Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Visualize final state
    visualize_final_state(sim)


def visualize_final_state(sim):
    """Visualize the final state of the simulation.
    
    Args:
        sim: The completed simulation
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the food grid
    bounds = sim.environment.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    grid_size = sim.environment.state["grid_size"]
    
    food_grid = sim.environment.get_food_grid()
    plt.imshow(food_grid.T, origin='lower', extent=bounds, 
               cmap='YlGn', alpha=0.5, interpolation='none')
    
    # Plot agents
    prey_x = []
    prey_y = []
    predator_x = []
    predator_y = []
    
    for agent in sim.agents:
        if agent.state.get("alive", False):
            if agent.state.get("type") == "prey":
                prey_x.append(agent.position[0])
                prey_y.append(agent.position[1])
            elif agent.state.get("type") == "predator":
                predator_x.append(agent.position[0])
                predator_y.append(agent.position[1])
    
    # Plot prey as blue dots
    plt.scatter(prey_x, prey_y, c='blue', label='Prey', s=30, alpha=0.7)
    
    # Plot predators as red triangles
    plt.scatter(predator_x, predator_y, c='red', label='Predators', 
                marker='^', s=50, alpha=0.7)
    
    plt.colorbar(label='Food Amount')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Ecosystem State')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Run a simulation and visualize the results
if __name__ == "__main__":
    print("Running predator-prey agent-based simulation...")
    
    # Run with default parameters
    simulation = run_predator_prey_simulation(
        num_prey=100,
        num_predators=20,
        days=200,
        environment_size=(0, 0, 100, 100),
        initial_food_density=0.3
    )
    
    # Print final statistics
    final_metrics = simulation.metrics[-1]
    print("\nSimulation Results:")
    print(f"Final prey population: {final_metrics['prey_count']}")
    print(f"Final predator population: {final_metrics['predator_count']}")
    print(f"Average prey energy: {final_metrics['avg_prey_energy']:.2f}")
    print(f"Average predator energy: {final_metrics['avg_predator_energy']:.2f}")
    print(f"Total prey born: {final_metrics['prey_born']}")
    print(f"Total predators born: {final_metrics['predator_born']}")
    
    # Visualize results
    visualize_simulation_results(simulation)