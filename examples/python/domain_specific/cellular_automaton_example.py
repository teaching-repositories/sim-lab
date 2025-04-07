"""
Example of using SimLab to create a cellular automaton simulation.

This example demonstrates:
1. Creating and running Conway's Game of Life
2. Implementing custom cellular automaton rules
3. Visualizing the evolution of cellular patterns
4. Detecting stable patterns and cycles
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import List


def create_glider(rows, cols, r_offset=0, c_offset=0):
    """Create a grid with a glider pattern.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        r_offset: Row offset for the glider position
        c_offset: Column offset for the glider position
    
    Returns:
        A grid with a glider pattern
    """
    grid = np.zeros((rows, cols), dtype=int)
    
    # Define glider pattern - a simple pattern that moves diagonally
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    
    # Position the glider in the grid with offsets
    r = r_offset
    c = c_offset
    grid[r:r+3, c:c+3] = glider
    
    return grid


def create_blinker(rows, cols, r_offset=0, c_offset=0):
    """Create a grid with a blinker pattern (oscillator with period 2).
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        r_offset: Row offset for the blinker position
        c_offset: Column offset for the blinker position
    
    Returns:
        A grid with a blinker pattern
    """
    grid = np.zeros((rows, cols), dtype=int)
    
    # Define blinker pattern - vertical line of 3 cells
    r = r_offset
    c = c_offset
    grid[r:r+3, c] = 1
    
    return grid


def create_random_grid(rows, cols, density=0.3, seed=None):
    """Create a random grid with the specified density of live cells.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        density: Proportion of cells that should be alive
        seed: Random seed for reproducibility
    
    Returns:
        A randomly populated grid
    """
    rng = np.random.RandomState(seed)
    return (rng.random((rows, cols)) < density).astype(int)


def custom_rule_elementary_ca(grid):
    """Implement an elementary cellular automaton (Rule 30).
    
    This implements Rule 30, where each cell looks at itself and its 
    immediate neighbors to determine its next state based on the rule.
    
    Args:
        grid: Current state of the grid
        
    Returns:
        Updated grid after applying the rule
    """
    # For elementary CA, we'll assume a 1D grid and only process the first row
    # and wrap around for boundary conditions
    if grid.shape[0] > 1:
        # We'll only use the first row for elementary CA
        row = grid[0, :].copy()
    else:
        row = grid.copy().flatten()
    
    # Pad the row for boundary calculations
    padded = np.pad(row, 1, mode='wrap')
    
    # Apply Rule 30: New cell state = left XOR (center OR right)
    new_row = np.zeros_like(row)
    for i in range(len(row)):
        left = padded[i]
        center = padded[i+1]
        right = padded[i+2]
        new_row[i] = (left ^ (center | right))
    
    # Create a new 2D grid where each row shows the evolution
    new_grid = np.zeros_like(grid)
    
    # Shift existing rows down by one
    if grid.shape[0] > 1:
        new_grid[1:, :] = grid[:-1, :]
    
    # Add new row at the top
    new_grid[0, :] = new_row
    
    return new_grid


def custom_rule_brain(grid):
    """Implement Brian's Brain cellular automaton.
    
    This is a cellular automaton with three states:
    0: Dead (off)
    1: Dying (just fired)
    2: Alive (on)
    
    Rules:
    - If a cell is off and has exactly 2 neighbors that are on, it turns on
    - If a cell is on, it becomes dying
    - If a cell is dying, it becomes off
    
    Args:
        grid: Current state of the grid
        
    Returns:
        Updated grid after applying the rule
    """
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    
    # Pad grid for boundary calculations
    padded = np.pad(grid, 1, mode='wrap')
    
    # Count neighbors that are in the "on" state (value 2)
    on_neighbors = np.zeros((rows, cols), dtype=int)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:  # Skip cell itself
                continue
            shifted = padded[i:i+rows, j:j+cols]
            on_neighbors += (shifted == 2).astype(int)
    
    # Apply rules
    # 1. Off cells with exactly 2 on neighbors become on
    new_grid[(grid == 0) & (on_neighbors == 2)] = 2
    
    # 2. On cells become dying
    new_grid[grid == 2] = 1
    
    # 3. Dying cells become off
    # (Already set to 0 initially)
    
    return new_grid


def run_game_of_life(initial_state=None, grid_size=(50, 50), days=100, density=0.3, boundary='periodic'):
    """Run Conway's Game of Life simulation.
    
    Args:
        initial_state: Initial configuration. If None, a random grid is generated.
        grid_size: Size of the grid as (rows, columns)
        days: Number of generations to simulate
        density: Density of live cells for random initialization
        boundary: Boundary condition ('periodic' or 'fixed')
        
    Returns:
        The completed simulation
    """
    # Create the simulation
    sim = SimulatorRegistry.create(
        "CellularAutomaton",
        grid_size=grid_size,
        initial_state=initial_state,
        initial_density=density,
        rule="game_of_life",
        days=days,
        boundary=boundary,
        random_seed=42
    )
    
    # Run the simulation
    live_cells = sim.run_simulation()
    
    return sim, live_cells


def run_custom_cellular_automaton(
    rule_function, initial_state=None, grid_size=(50, 50), days=100, density=0.3
):
    """Run a cellular automaton with a custom rule.
    
    Args:
        rule_function: Custom function to update the grid
        initial_state: Initial configuration. If None, a random grid is generated.
        grid_size: Size of the grid as (rows, columns)
        days: Number of generations to simulate
        density: Density of live cells for random initialization
        
    Returns:
        The completed simulation
    """
    # Create the simulation
    sim = SimulatorRegistry.create(
        "CellularAutomaton",
        grid_size=grid_size,
        initial_state=initial_state,
        initial_density=density,
        rule=rule_function,
        days=days,
        random_seed=42
    )
    
    # Run the simulation
    live_cells = sim.run_simulation()
    
    return sim, live_cells


def visualize_grid_evolution(sim, frames=None, interval=200):
    """Visualize the evolution of the cellular automaton grid over time.
    
    Args:
        sim: The completed simulation
        frames: Number of frames to show (defaults to all states)
        interval: Time between animation frames in milliseconds
    """
    states = sim.get_all_states()
    
    if frames is None:
        frames = len(states)
    else:
        frames = min(frames, len(states))
    
    # Create the figure and initial plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Cellular Automaton - Generation 0')
    
    # Initial state
    img = ax.imshow(states[0], cmap='binary', interpolation='nearest')
    
    def update(frame):
        """Update function for animation."""
        img.set_array(states[frame])
        ax.set_title(f'Cellular Automaton - Generation {frame}')
        return [img]
    
    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )
    
    plt.tight_layout()
    plt.show()


def plot_live_cells_over_time(live_cells, title="Live Cells Over Time"):
    """Plot the number of live cells over time.
    
    Args:
        live_cells: List of live cell counts for each generation
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(live_cells, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Number of Live Cells')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def detect_and_display_patterns(sim):
    """Detect and display any stable patterns or cycles in the simulation.
    
    Args:
        sim: The completed simulation
    """
    cycle_length = sim.detect_stable_pattern(max_cycle_length=20)
    
    if cycle_length is None:
        print("No stable pattern detected.")
    elif cycle_length == 0:
        print("Simulation reached a stable state.")
        
        # Show the final stable state
        plt.figure(figsize=(8, 8))
        plt.imshow(sim.get_state_at_day(-1), cmap='binary', interpolation='nearest')
        plt.title('Stable Pattern')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Detected a cycle with period {cycle_length}.")
        
        # Show the cycle
        cycle_states = sim.get_all_states()[-cycle_length:]
        
        fig, axes = plt.subplots(1, cycle_length, figsize=(3 * cycle_length, 3))
        if cycle_length == 1:
            axes = [axes]  # Handle case of a single subplot
        
        for i, state in enumerate(cycle_states):
            axes[i].imshow(state, cmap='binary', interpolation='nearest')
            axes[i].set_title(f'Step {i + 1}')
            axes[i].axis('off')
        
        plt.suptitle(f'Cycle with Period {cycle_length}')
        plt.tight_layout()
        plt.show()


# Run different cellular automaton simulations
if __name__ == "__main__":
    print("Running cellular automaton simulations...")
    
    # 1. Conway's Game of Life with a glider
    print("\n1. Conway's Game of Life with a glider")
    grid_size = (30, 30)
    glider_grid = create_glider(grid_size[0], grid_size[1], 5, 5)
    gol_sim, gol_live_cells = run_game_of_life(
        initial_state=glider_grid,
        grid_size=grid_size,
        days=50,
        boundary='periodic'
    )
    
    print(f"Final number of live cells: {gol_live_cells[-1]}")
    print("Visualizing glider evolution...")
    visualize_grid_evolution(gol_sim, frames=50, interval=150)
    
    # 2. Conway's Game of Life with a blinker oscillator
    print("\n2. Conway's Game of Life with a blinker oscillator")
    blinker_grid = create_blinker(grid_size[0], grid_size[1], 10, 15)
    blinker_sim, blinker_live_cells = run_game_of_life(
        initial_state=blinker_grid,
        grid_size=grid_size,
        days=10
    )
    
    print("Detecting patterns...")
    detect_and_display_patterns(blinker_sim)
    
    # 3. Elementary cellular automaton (Rule 30)
    print("\n3. Elementary Cellular Automaton (Rule 30)")
    # Create a grid with just one cell on in the middle of the top row
    elementary_grid = np.zeros((50, 100), dtype=int)
    elementary_grid[0, 50] = 1
    
    elementary_sim, elementary_live_cells = run_custom_cellular_automaton(
        rule_function=custom_rule_elementary_ca,
        initial_state=elementary_grid,
        grid_size=(50, 100),
        days=50
    )
    
    print("Visualizing Rule 30 evolution...")
    visualize_grid_evolution(elementary_sim, frames=50, interval=100)
    
    # 4. Brian's Brain cellular automaton
    print("\n4. Brian's Brain Cellular Automaton")
    # Create a random 3-state grid
    brain_grid = np.random.randint(0, 3, size=(50, 50))
    
    brain_sim, brain_live_cells = run_custom_cellular_automaton(
        rule_function=custom_rule_brain,
        initial_state=brain_grid,
        grid_size=(50, 50),
        days=100
    )
    
    print("Visualizing Brian's Brain evolution...")
    visualize_grid_evolution(brain_sim, frames=50, interval=100)
    
    # Plot the number of live cells over time for Conway's Game of Life
    plot_live_cells_over_time(gol_live_cells, "Live Cells in Conway's Game of Life")