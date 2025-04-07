"""
Example of using SimLab to create epidemiological simulations.

This example demonstrates:
1. Setting up and running a basic SIR epidemic model
2. Analyzing the impact of different disease parameters
3. Simulating intervention strategies
4. Visualizing epidemic curves and reproduction numbers
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def run_sir_model(
    population_size=10000,
    initial_infected=10,
    initial_recovered=0,
    beta=0.3,
    gamma=0.1,
    days=150
):
    """Run a basic SIR (Susceptible, Infected, Recovered) model.
    
    Args:
        population_size: Total population size
        initial_infected: Initial number of infected individuals
        initial_recovered: Initial number of recovered individuals
        beta: Transmission rate (rate at which susceptible individuals become infected)
        gamma: Recovery rate (rate at which infected individuals recover)
        days: Number of days to simulate
        
    Returns:
        The completed simulation
    """
    # Create the simulation
    sim = SimulatorRegistry.create(
        "Epidemiological",
        population_size=population_size,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        beta=beta,
        gamma=gamma,
        days=days,
        random_seed=42
    )
    
    # Run the simulation
    sim.run_simulation()
    
    return sim


def visualize_epidemic_curve(sim):
    """Visualize the SIR epidemic curve.
    
    Args:
        sim: The completed simulation
    """
    # Get compartment data
    compartments = sim.get_compartments()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each compartment
    days = range(len(compartments['susceptible']))
    plt.plot(days, compartments['susceptible'], 'b-', label='Susceptible', linewidth=2)
    plt.plot(days, compartments['infected'], 'r-', label='Infected', linewidth=2)
    plt.plot(days, compartments['recovered'], 'g-', label='Recovered', linewidth=2)
    
    # Mark the peak of the epidemic
    peak_day, peak_value = sim.get_peak_infection()
    plt.scatter(peak_day, peak_value, color='black', s=100, zorder=5)
    plt.annotate(f'Peak: Day {peak_day}, {peak_value:.0f} cases',
                xy=(peak_day, peak_value),
                xytext=(peak_day + 10, peak_value + 500),
                arrowprops=dict(arrowstyle='->'))
    
    # Add labels and legend
    plt.title('SIR Epidemic Model')
    plt.xlabel('Days')
    plt.ylabel('Number of Individuals')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add R0 information
    r0 = sim.get_reproduction_number()
    plt.figtext(0.15, 0.85, f'Basic Reproduction Number (R0): {r0:.2f}',
               fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add final statistics
    final_sizes = sim.get_final_sizes()
    plt.figtext(0.15, 0.80, 
               f'Final statistics:\n'
               f'- Infected: {final_sizes["infected"]:.0f} ({final_sizes["infected"]/sim.population_size:.1%})\n'
               f'- Recovered: {final_sizes["recovered"]:.0f} ({final_sizes["recovered"]/sim.population_size:.1%})\n'
               f'- Susceptible: {final_sizes["susceptible"]:.0f} ({final_sizes["susceptible"]/sim.population_size:.1%})',
               fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def compare_reproduction_numbers():
    """Compare epidemic curves for different reproduction numbers."""
    # Define different R0 values to compare
    r0_values = [1.5, 2.5, 3.5]
    gamma = 0.1  # Fixed recovery rate
    
    # Run simulations for each R0
    simulations = []
    for r0 in r0_values:
        # Calculate beta to achieve the desired R0
        beta = r0 * gamma
        
        # Run simulation
        sim = run_sir_model(
            population_size=10000,
            initial_infected=10,
            beta=beta,
            gamma=gamma,
            days=200
        )
        
        simulations.append((r0, sim))
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot infected curves for each R0
    for r0, sim in simulations:
        compartments = sim.get_compartments()
        days = range(len(compartments['infected']))
        plt.plot(days, compartments['infected'], label=f'R0 = {r0}', linewidth=2)
        
        # Mark the peaks
        peak_day, peak_value = sim.get_peak_infection()
        plt.scatter(peak_day, peak_value, s=50)
    
    # Add labels and legend
    plt.title('Effect of Reproduction Number (R0) on Epidemic Curve')
    plt.xlabel('Days')
    plt.ylabel('Number of Infected Individuals')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compare final sizes
    plt.figure(figsize=(10, 6))
    
    # Extract data
    r0_list = [r0 for r0, _ in simulations]
    final_infected = [sim.get_final_sizes()['infected'] for _, sim in simulations]
    final_recovered = [sim.get_final_sizes()['recovered'] for _, sim in simulations]
    final_susceptible = [sim.get_final_sizes()['susceptible'] for _, sim in simulations]
    
    # Normalize to percentages
    population = simulations[0][1].population_size
    final_infected_pct = [100 * i / population for i in final_infected]
    final_recovered_pct = [100 * r / population for r in final_recovered]
    final_susceptible_pct = [100 * s / population for s in final_susceptible]
    
    # Create stacked bar chart
    width = 0.35
    plt.bar(r0_list, final_susceptible_pct, width, label='Susceptible', color='blue')
    plt.bar(r0_list, final_recovered_pct, width, bottom=final_susceptible_pct, 
            label='Recovered', color='green')
    plt.bar(r0_list, final_infected_pct, width, 
            bottom=[s+r for s, r in zip(final_susceptible_pct, final_recovered_pct)], 
            label='Infected', color='red')
    
    # Add labels and legend
    plt.title('Final Epidemic Outcome by Reproduction Number')
    plt.xlabel('Reproduction Number (R0)')
    plt.ylabel('Percentage of Population')
    plt.xticks(r0_list)
    plt.legend()
    
    # Add text annotations
    for i, r0 in enumerate(r0_list):
        recovered = final_recovered_pct[i]
        plt.text(r0, recovered/2 + final_susceptible_pct[i], f'{recovered:.1f}%', 
                 ha='center', va='center', color='white', fontweight='bold')
        
        susceptible = final_susceptible_pct[i]
        plt.text(r0, susceptible/2, f'{susceptible:.1f}%', 
                 ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Compute and display summary statistics
    print("\nComparison of epidemics with different R0 values:")
    for r0, sim in simulations:
        peak_day, peak_value = sim.get_peak_infection()
        final_sizes = sim.get_final_sizes()
        
        print(f"\nR0 = {r0}:")
        print(f"  Peak day: {peak_day}")
        print(f"  Peak infected: {peak_value:.0f} ({peak_value/sim.population_size:.1%} of population)")
        print(f"  Final recovered: {final_sizes['recovered']:.0f} ({final_sizes['recovered']/sim.population_size:.1%} of population)")
        print(f"  Remained susceptible: {final_sizes['susceptible']:.0f} ({final_sizes['susceptible']/sim.population_size:.1%} of population)")


def simulate_intervention_strategies():
    """Simulate different intervention strategies to control an epidemic."""
    # Base parameters
    population_size = 10000
    initial_infected = 10
    baseline_beta = 0.3
    gamma = 0.1
    days = 200
    
    # Define intervention scenarios
    scenarios = [
        {
            "name": "No intervention",
            "description": "Baseline scenario with no interventions",
            "beta_changes": []  # No changes to beta
        },
        {
            "name": "Early lockdown",
            "description": "Strict lockdown implemented early, then gradual reopening",
            "beta_changes": [
                (20, 0.05),    # Day 20: Strong lockdown reduces beta to 0.05
                (50, 0.15),    # Day 50: Partial reopening
                (80, 0.25)     # Day 80: Further reopening
            ]
        },
        {
            "name": "Delayed response",
            "description": "Interventions implemented only after cases rise significantly",
            "beta_changes": [
                (60, 0.10),    # Day 60: Strong lockdown
                (100, 0.20)    # Day 100: Partial reopening
            ]
        },
        {
            "name": "Cyclic measures",
            "description": "Implementing and relaxing measures in cycles",
            "beta_changes": [
                (30, 0.10),    # Day 30: Strong measures
                (60, 0.25),    # Day 60: Relaxation
                (90, 0.10),    # Day 90: Strong measures again
                (120, 0.25)    # Day 120: Relaxation again
            ]
        }
    ]
    
    # Run simulations for each scenario
    simulation_results = []
    
    for scenario in scenarios:
        # Create a new simulation
        sim = SimulatorRegistry.create(
            "Epidemiological",
            population_size=population_size,
            initial_infected=initial_infected,
            beta=baseline_beta,
            gamma=gamma,
            days=days,
            random_seed=42
        )
        
        # Run simulation
        infected = sim.run_simulation()
        
        # Store initial beta history
        beta_history = [baseline_beta] * days
        
        # Implement beta changes according to the scenario
        compartments = sim.get_compartments()
        susceptible = compartments['susceptible'].copy()
        infected = compartments['infected'].copy()
        recovered = compartments['recovered'].copy()
        
        for day, new_beta in scenario["beta_changes"]:
            # Update beta from this day forward
            beta_history[day:] = [new_beta] * (days - day)
            
            # Recalculate the epidemic from this point
            for t in range(day, days - 1):
                # Update with the new beta
                s, i, r = susceptible[t], infected[t], recovered[t]
                
                # Calculate new values using the updated beta
                new_infections = beta_history[t] * s * i / population_size
                new_recoveries = gamma * i
                
                susceptible[t+1] = s - new_infections
                infected[t+1] = i + new_infections - new_recoveries
                recovered[t+1] = r + new_recoveries
        
        # Create updated results dictionary
        updated_results = {
            "name": scenario["name"],
            "description": scenario["description"],
            "beta_history": beta_history,
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered
        }
        
        simulation_results.append(updated_results)
    
    # Visualize infected curves for all scenarios
    plt.figure(figsize=(12, 6))
    
    for result in simulation_results:
        plt.plot(range(days), result["infected"], 
                 label=result["name"], linewidth=2)
    
    plt.title('Effect of Different Intervention Strategies')
    plt.xlabel('Days')
    plt.ylabel('Number of Infected Individuals')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot beta values over time for each scenario
    plt.figure(figsize=(12, 5))
    
    for result in simulation_results:
        plt.plot(range(days), result["beta_history"], 
                 label=result["name"], linewidth=2)
    
    plt.title('Transmission Rate (β) Under Different Intervention Strategies')
    plt.xlabel('Days')
    plt.ylabel('Transmission Rate (β)')
    plt.ylim(0, baseline_beta * 1.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compare final outcomes
    final_infected = [result["infected"][-1] for result in simulation_results]
    final_recovered = [result["recovered"][-1] for result in simulation_results]
    final_susceptible = [result["susceptible"][-1] for result in simulation_results]
    
    # Calculate total cases (recovered + currently infected)
    total_cases = [i + r for i, r in zip(final_infected, final_recovered)]
    
    # Calculate peak infected
    peak_infected = [max(result["infected"]) for result in simulation_results]
    peak_days = [result["infected"].index(max(result["infected"])) for result in simulation_results]
    
    # Print summary
    print("\nComparison of intervention strategies:")
    for i, result in enumerate(simulation_results):
        print(f"\n{result['name']}:")
        print(f"  Description: {result['description']}")
        print(f"  Peak infected: {peak_infected[i]:.0f} on day {peak_days[i]}")
        print(f"  Total cases: {total_cases[i]:.0f} ({total_cases[i]/population_size:.1%} of population)")
        print(f"  Final infected: {final_infected[i]:.0f}")
        print(f"  Final recovered: {final_recovered[i]:.0f}")
        print(f"  Remained susceptible: {final_susceptible[i]:.0f} ({final_susceptible[i]/population_size:.1%} of population)")


def analyze_herd_immunity_threshold():
    """Analyze the herd immunity threshold for different R0 values."""
    # Define R0 values to analyze
    r0_values = np.arange(1.0, 5.1, 0.5)
    
    # Calculate theoretical herd immunity threshold: 1 - 1/R0
    herd_immunity_thresholds = [1 - 1/r0 for r0 in r0_values]
    
    # Simulate to verify the theoretical threshold
    gamma = 0.1  # Fixed recovery rate
    population_size = 10000
    
    # For each R0, run simulations with different initial recovered percentages
    results = []
    
    for r0 in [1.5, 2.5, 3.5]:  # Subset for detailed analysis
        beta = r0 * gamma
        theoretical_threshold = 1 - 1/r0
        
        # Test different initial immunity levels
        immunity_levels = np.linspace(0, 0.95, 20)
        outbreak_sizes = []
        
        for immunity in immunity_levels:
            # Calculate initial compartment sizes
            initial_recovered = int(immunity * population_size)
            initial_infected = 10
            initial_susceptible = population_size - initial_recovered - initial_infected
            
            # Run simulation
            sim = SimulatorRegistry.create(
                "Epidemiological",
                population_size=population_size,
                initial_infected=initial_infected,
                initial_recovered=initial_recovered,
                beta=beta,
                gamma=gamma,
                days=200,
                random_seed=42
            )
            
            sim.run_simulation()
            
            # Get size of the outbreak (total new cases)
            final_sizes = sim.get_final_sizes()
            total_cases = final_sizes['infected'] + final_sizes['recovered'] - initial_recovered
            outbreak_sizes.append(total_cases)
        
        results.append({
            "r0": r0,
            "theoretical_threshold": theoretical_threshold,
            "immunity_levels": immunity_levels,
            "outbreak_sizes": outbreak_sizes
        })
    
    # Plot theoretical herd immunity threshold
    plt.figure(figsize=(10, 6))
    plt.plot(r0_values, herd_immunity_thresholds, 'r-', linewidth=2)
    plt.fill_between(r0_values, herd_immunity_thresholds, 1, alpha=0.2, color='red',
                     label='Herd immunity achieved')
    plt.fill_between(r0_values, 0, herd_immunity_thresholds, alpha=0.2, color='blue',
                     label='Susceptible to outbreaks')
    
    # Add annotations for common diseases
    diseases = [
        (1.5, "Seasonal Flu (R0 ≈ 1.5)"),
        (2.5, "SARS (R0 ≈ 2.5)"),
        (3.0, "COVID-19 (R0 ≈ 3.0)"),
        (5.0, "Measles (R0 ≈ 15+)")
    ]
    
    for r0, label in diseases:
        if r0 <= max(r0_values):
            thresh = 1 - 1/r0
            plt.scatter([r0], [thresh], s=100, zorder=5)
            plt.annotate(label, xy=(r0, thresh), xytext=(r0 + 0.1, thresh + 0.05))
    
    plt.title('Herd Immunity Threshold by Reproduction Number')
    plt.xlabel('Basic Reproduction Number (R0)')
    plt.ylabel('Proportion of Immune Population Required')
    plt.xlim(1, max(r0_values))
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot detailed simulation results showing outbreak sizes
    plt.figure(figsize=(12, 8))
    
    for i, result in enumerate(results):
        plt.subplot(len(results), 1, i+1)
        
        r0 = result["r0"]
        theoretical_threshold = result["theoretical_threshold"]
        
        # Plot outbreak sizes
        plt.plot(result["immunity_levels"] * 100, result["outbreak_sizes"], 
                 'b-', linewidth=2, label=f'Outbreak size (R0 = {r0})')
        
        # Mark theoretical threshold
        plt.axvline(x=theoretical_threshold * 100, color='r', linestyle='--', 
                    label=f'Theoretical threshold: {theoretical_threshold:.1%}')
        
        # Add labels
        plt.title(f'Outbreak Size vs. Initial Immunity Level (R0 = {r0})')
        plt.xlabel('Percentage of Initially Immune Population')
        plt.ylabel('Outbreak Size (Total New Cases)')
        plt.grid(alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()


# Run various epidemiological simulations
if __name__ == "__main__":
    # 1. Basic SIR model
    print("1. Running basic SIR epidemic model...")
    sir_sim = run_sir_model()
    visualize_epidemic_curve(sir_sim)
    
    # Calculate and display basic statistics
    peak_day, peak_value = sir_sim.get_peak_infection()
    final_sizes = sir_sim.get_final_sizes()
    r0 = sir_sim.get_reproduction_number()
    
    print(f"Basic SIR Model Results:")
    print(f"  Basic Reproduction Number (R0): {r0}")
    print(f"  Peak of epidemic: Day {peak_day}, {peak_value:.0f} cases")
    print(f"  Final epidemic size: {final_sizes['recovered']:.0f} recovered individuals")
    print(f"  Attack rate: {final_sizes['recovered']/sir_sim.population_size:.1%}")
    
    # 2. Compare different reproduction numbers
    print("\n2. Comparing different reproduction numbers...")
    compare_reproduction_numbers()
    
    # 3. Simulate intervention strategies
    print("\n3. Simulating intervention strategies...")
    simulate_intervention_strategies()
    
    # 4. Analyze herd immunity threshold
    print("\n4. Analyzing herd immunity threshold...")
    analyze_herd_immunity_threshold()