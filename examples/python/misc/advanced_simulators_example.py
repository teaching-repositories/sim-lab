"""
Example demonstrating the use of advanced simulators.

This script shows how to use:
1. Monte Carlo Simulation
2. Epidemiological (SIR) Simulation
3. Cellular Automaton Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

from sim_lab.core import (
    SimulatorRegistry,
    MonteCarloSimulation,
    EpidemiologicalSimulation, 
    CellularAutomatonSimulation
)


def run_monte_carlo_example():
    """Run a Monte Carlo simulation example for estimating Pi."""
    print("\n--- Monte Carlo Simulation Example: Estimating Pi ---")
    
    # Define sample and evaluation functions for estimating Pi
    def sample_point():
        """Generate a random point in the unit square."""
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        return (x, y)
    
    def is_in_circle(point):
        """Check if a point is inside the unit circle (return 1 if yes, 0 if no)."""
        x, y = point
        return 1.0 if x**2 + y**2 <= 1 else 0.0
    
    # Create the simulation
    mc_sim = MonteCarloSimulation(
        sample_function=sample_point,
        evaluation_function=is_in_circle,
        num_samples=1000,  # Number of random points per step
        days=100,          # Number of steps
        random_seed=42
    )
    
    # Run the simulation
    results = mc_sim.run_simulation()
    
    # Calculate Pi estimates (points in circle / total points * 4)
    pi_estimates = [r * 4 for r in results]
    
    # Get confidence intervals
    confidence_intervals = mc_sim.get_confidence_intervals()
    ci_lower = [ci[0] * 4 for ci in confidence_intervals]
    ci_upper = [ci[1] * 4 for ci in confidence_intervals]
    
    # Get statistics
    stats = mc_sim.get_statistics()
    print(f"Final Pi estimate: {pi_estimates[-1]:.6f}")
    print(f"True Pi value: {np.pi:.6f}")
    print(f"Error: {abs(pi_estimates[-1] - np.pi):.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(pi_estimates, label='Pi Estimate')
    plt.axhline(y=np.pi, color='r', linestyle='-', label='True Pi Value')
    plt.fill_between(range(len(ci_lower)), ci_lower, ci_upper, alpha=0.2, color='blue')
    plt.title('Monte Carlo Estimation of Pi')
    plt.xlabel('Iteration')
    plt.ylabel('Pi Estimate')
    plt.grid(True)
    plt.legend()
    plt.savefig("monte_carlo_pi.png")
    print("Saved Monte Carlo results to 'monte_carlo_pi.png'")


def run_epidemiological_example():
    """Run an epidemiological (SIR) simulation example."""
    print("\n--- Epidemiological Simulation Example: SIR Model ---")
    
    # Create the SIR simulation
    pop_size = 10000
    epi_sim = EpidemiologicalSimulation(
        population_size=pop_size,
        initial_infected=10,
        initial_recovered=0,
        beta=0.3,              # Infection rate
        gamma=0.1,             # Recovery rate
        days=200,
        random_seed=42
    )
    
    # Run the simulation
    infected_over_time = epi_sim.run_simulation()
    
    # Get all compartments data
    compartments = epi_sim.get_compartments()
    
    # Get peak infection info
    peak_day, peak_value = epi_sim.get_peak_infection()
    print(f"Peak infection occurs on day {peak_day} with {peak_value:.0f} infected individuals")
    
    # Get R0 value
    r0 = epi_sim.get_reproduction_number()
    print(f"Basic reproduction number (R0): {r0:.2f}")
    
    # Get final sizes
    final_sizes = epi_sim.get_final_sizes()
    print(f"Final state: {final_sizes['susceptible']:.0f} susceptible, " 
          f"{final_sizes['infected']:.0f} infected, {final_sizes['recovered']:.0f} recovered")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(compartments['susceptible'], label='Susceptible', color='blue')
    plt.plot(compartments['infected'], label='Infected', color='red')
    plt.plot(compartments['recovered'], label='Recovered', color='green')
    plt.axvline(x=peak_day, color='black', linestyle='--', alpha=0.5, label=f'Peak on day {peak_day}')
    plt.title('SIR Epidemiological Model')
    plt.xlabel('Day')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.grid(True)
    plt.savefig("sir_model.png")
    print("Saved SIR model results to 'sir_model.png'")


def run_cellular_automaton_example():
    """Run a cellular automaton simulation example (Conway's Game of Life)."""
    print("\n--- Cellular Automaton Simulation Example: Conway's Game of Life ---")
    
    # Create the Game of Life simulation with a glider pattern
    grid_size = (50, 50)
    initial_state = np.zeros(grid_size, dtype=int)
    
    # Add a glider pattern
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    initial_state[5:8, 5:8] = glider
    
    # Add a blinker pattern
    blinker = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    initial_state[20:23, 20:23] = blinker
    
    # Create simulation
    ca_sim = CellularAutomatonSimulation(
        grid_size=grid_size,
        initial_state=initial_state,
        rule="game_of_life",
        days=100,
        boundary='periodic',
        random_seed=42
    )
    
    # Run the simulation
    live_cells = ca_sim.run_simulation()
    
    # Check for stable patterns
    stability = ca_sim.detect_stable_pattern()
    if stability is not None:
        if stability == 0:
            print("Simulation reached a stable state")
        else:
            print(f"Simulation reached a cycle with period {stability}")
    
    # Plot initial, middle, and final states
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(ca_sim.get_state_at_day(0), cmap='binary')
    axs[0].set_title('Initial State')
    axs[0].axis('off')
    
    middle_day = ca_sim.days // 2
    axs[1].imshow(ca_sim.get_state_at_day(middle_day), cmap='binary')
    axs[1].set_title(f'Day {middle_day}')
    axs[1].axis('off')
    
    axs[2].imshow(ca_sim.get_state_at_day(ca_sim.days - 1), cmap='binary')
    axs[2].set_title(f'Final State (Day {ca_sim.days - 1})')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("game_of_life.png")
    print("Saved Game of Life results to 'game_of_life.png'")
    
    # Plot the number of live cells over time
    plt.figure(figsize=(10, 6))
    plt.plot(live_cells)
    plt.title('Number of Live Cells Over Time')
    plt.xlabel('Generation')
    plt.ylabel('Live Cells')
    plt.grid(True)
    plt.savefig("live_cells.png")
    print("Saved live cells count to 'live_cells.png'")


def main():
    """Run all advanced simulator examples."""
    # List all available simulators
    print("Available simulators:")
    for sim_name in SimulatorRegistry.list_simulators():
        print(f"  - {sim_name}")
    
    # Run the examples
    run_monte_carlo_example()
    run_epidemiological_example()
    run_cellular_automaton_example()


if __name__ == "__main__":
    main()