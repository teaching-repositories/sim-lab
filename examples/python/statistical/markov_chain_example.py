"""
Example of using SimLab to create Markov chain simulations.

This example demonstrates:
1. Creating and running Markov chain simulations
2. Working with weather forecast models
3. Simulating random walks with boundary conditions
4. Modeling inventory management systems
5. Analyzing state distributions and transitions
"""

from sim_lab.core import SimulatorRegistry, create_weather_model, create_random_walk, create_inventory_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any
from matplotlib.colors import LinearSegmentedColormap


def run_weather_simulation():
    """Run a weather forecast simulation using a Markov chain.
    
    This example models daily weather transitions between sunny, cloudy, and rainy
    states based on a transition probability matrix.
    """
    print("Running weather forecast simulation...")
    
    # Create a weather model with custom parameters
    sim = create_weather_model(
        sunny_to_sunny=0.7,
        sunny_to_cloudy=0.2,
        sunny_to_rainy=0.1,
        cloudy_to_sunny=0.3,
        cloudy_to_cloudy=0.4,
        cloudy_to_rainy=0.3,
        rainy_to_sunny=0.2,
        rainy_to_cloudy=0.3,
        rainy_to_rainy=0.5,
        initial_state="Sunny",
        days=50
    )
    
    # Alternatively, create the model using SimulatorRegistry
    # transition_matrix = np.array([
    #     [0.7, 0.2, 0.1],  # Sunny -> Sunny, Cloudy, Rainy
    #     [0.3, 0.4, 0.3],  # Cloudy -> Sunny, Cloudy, Rainy
    #     [0.2, 0.3, 0.5]   # Rainy -> Sunny, Cloudy, Rainy
    # ])
    # sim = SimulatorRegistry.create(
    #     "MarkovChain",
    #     transition_matrix=transition_matrix,
    #     states=["Sunny", "Cloudy", "Rainy"],
    #     initial_state="Sunny",
    #     days=50
    # )
    
    # Run the simulation
    sim.run_simulation()
    
    # Get the sequence of weather states
    weather_sequence = sim.get_state_names()
    
    # Print a summary of the first few days
    print("\nWeather forecast for the first 10 days:")
    for day, weather in enumerate(weather_sequence[:10]):
        print(f"Day {day+1}: {weather}")
    
    # Get the distribution of weather states
    state_distribution = sim.get_state_distribution()
    print("\nOverall weather distribution:")
    for state, frequency in state_distribution.items():
        print(f"{state}: {frequency:.2%}")
    
    # Compute the stationary distribution
    stationary_distribution = sim.compute_stationary_distribution()
    print("\nStationary distribution (long-term probabilities):")
    for i, state in enumerate(sim.states):
        print(f"{state}: {stationary_distribution[i]:.2%}")
    
    # Visualize the weather sequence
    plt.figure(figsize=(12, 6))
    
    # Map weather states to colors
    weather_colors = {
        "Sunny": "#FFD700",    # Gold
        "Cloudy": "#B0C4DE",   # Light steel blue
        "Rainy": "#4682B4"     # Steel blue
    }
    
    # Create a color sequence
    colors = [weather_colors[weather] for weather in weather_sequence]
    
    # Create the timeline plot
    for i, (weather, color) in enumerate(zip(weather_sequence, colors)):
        plt.fill_between([i, i+1], 0, 1, color=color)
        
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=weather_colors[state], label=state)
        for state in weather_colors
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlim(0, len(weather_sequence))
    plt.ylim(0, 1)
    plt.title('Daily Weather Simulation')
    plt.xlabel('Day')
    plt.yticks([])  # Hide y-axis ticks
    plt.tight_layout()
    plt.show()
    
    # Visualize the transition matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim.transition_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=sim.states,
        yticklabels=sim.states
    )
    plt.title('Weather Transition Probabilities')
    plt.xlabel('Next Day\'s Weather')
    plt.ylabel('Current Day\'s Weather')
    plt.tight_layout()
    plt.show()
    
    # Forecast weather probabilities for the next 7 days
    # starting from different initial conditions
    initial_states = ["Sunny", "Cloudy", "Rainy"]
    days_to_forecast = 7
    
    plt.figure(figsize=(15, 5))
    
    for idx, initial in enumerate(initial_states):
        # Create a new simulation with this initial state
        forecast_sim = create_weather_model(
            initial_state=initial,
            days=1  # We only need the initial state
        )
        
        # Get state probabilities over time
        state_probs = []
        forecast_sim.current_state = forecast_sim.state_to_index[initial]
        
        for day in range(days_to_forecast):
            probs = forecast_sim.predict_state_probabilities(day + 1)
            state_probs.append(probs)
        
        # Plot as subplots
        plt.subplot(1, 3, idx + 1)
        for i, state in enumerate(forecast_sim.states):
            probabilities = [probs[i] for probs in state_probs]
            plt.plot(range(1, days_to_forecast + 1), probabilities,
                     marker='o', linewidth=2, label=state)
        
        plt.title(f'Weather Forecast Starting from {initial}')
        plt.xlabel('Days from Now')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sim


def run_random_walk_simulation():
    """Run a random walk simulation using a Markov chain.
    
    This example models a random walk where a particle moves up or down
    with certain probabilities, bounded by minimum and maximum positions.
    """
    print("\nRunning random walk simulation...")
    
    # Create a random walk model
    sim = create_random_walk(
        p_up=0.55,    # Slightly biased upward
        p_down=0.45,
        initial_position=0,
        min_position=-10,
        max_position=10,
        days=200
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Get the sequence of positions
    position_sequence = sim.get_state_names()
    
    # Print a summary
    print(f"Starting position: {position_sequence[0]}")
    print(f"Final position: {position_sequence[-1]}")
    print(f"Minimum position reached: {min(position_sequence)}")
    print(f"Maximum position reached: {max(position_sequence)}")
    
    # Count how many times the walk hit the boundaries
    min_hits = position_sequence.count(min(sim.states))
    max_hits = position_sequence.count(max(sim.states))
    print(f"Number of times at minimum boundary: {min_hits}")
    print(f"Number of times at maximum boundary: {max_hits}")
    
    # Visualize the random walk
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(position_sequence)), position_sequence, 'b-')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark hitting boundaries
    min_bound = min(sim.states)
    max_bound = max(sim.states)
    min_indices = [i for i, pos in enumerate(position_sequence) if pos == min_bound]
    max_indices = [i for i, pos in enumerate(position_sequence) if pos == max_bound]
    
    plt.scatter(min_indices, [min_bound] * len(min_indices), color='red', s=20, label='Hit Minimum')
    plt.scatter(max_indices, [max_bound] * len(max_indices), color='green', s=20, label='Hit Maximum')
    
    plt.title('Random Walk Simulation')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Compute and display the state distribution
    state_distribution = sim.get_state_distribution()
    
    plt.figure(figsize=(10, 6))
    states = sorted(state_distribution.keys())
    frequencies = [state_distribution[state] for state in states]
    
    plt.bar(states, frequencies, color='skyblue')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Position Distribution in Random Walk')
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Run multiple simulations to analyze the distribution
    num_simulations = 1000
    final_positions = []
    
    for _ in range(num_simulations):
        sim_run = create_random_walk(
            p_up=0.55,
            p_down=0.45,
            initial_position=0,
            min_position=-10,
            max_position=10,
            days=100
        )
        sim_run.run_simulation()
        final_positions.append(sim_run.current_state)
    
    # Convert to actual positions
    final_positions = [sim.states[pos] for pos in final_positions]
    
    # Plot the distribution of final positions
    plt.figure(figsize=(10, 6))
    plt.hist(final_positions, bins=range(min(sim.states)-1, max(sim.states)+2), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Starting Position')
    
    plt.title(f'Distribution of Final Positions After {sim.days} Steps\n(from {num_simulations} simulations)')
    plt.xlabel('Final Position')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return sim


def run_inventory_simulation():
    """Run an inventory management simulation using a Markov chain.
    
    This example models a simple inventory system where:
    - The state is the current inventory level
    - Random demand reduces inventory
    - When inventory reaches 0, a new order is placed
    """
    print("\nRunning inventory management simulation...")
    
    # Create an inventory model with custom parameters
    sim = create_inventory_model(
        demand_probs=[0.2, 0.3, 0.3, 0.15, 0.05],  # Probabilities for demands of 0, 1, 2, 3, 4
        max_inventory=10,
        order_amount=7,
        days=100
    )
    
    # Run the simulation
    sim.run_simulation()
    
    # Get the sequence of inventory levels
    inventory_sequence = sim.get_state_names()
    
    # Count stock-outs (inventory = 0)
    stockouts = inventory_sequence.count(0)
    stockout_rate = stockouts / len(inventory_sequence)
    
    # Print summary statistics
    print(f"Initial inventory: {inventory_sequence[0]}")
    print(f"Final inventory: {inventory_sequence[-1]}")
    print(f"Average inventory level: {sum(inventory_sequence) / len(inventory_sequence):.2f}")
    print(f"Stockout rate: {stockout_rate:.2%}")
    print(f"Number of reorders: {stockouts}")
    
    # Visualize inventory levels over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(inventory_sequence)), inventory_sequence, 'b-')
    
    # Mark stockouts
    stockout_indices = [i for i, level in enumerate(inventory_sequence) if level == 0]
    if stockout_indices:
        plt.scatter(stockout_indices, [0] * len(stockout_indices), color='red', s=30, label='Stockout')
    
    # Mark reorder points (inventory = 0 followed by order_amount)
    reorder_indices = []
    for i in range(len(inventory_sequence) - 1):
        if inventory_sequence[i] == 0 and inventory_sequence[i+1] == sim.order_amount:
            reorder_indices.append(i)
    
    if reorder_indices:
        plt.scatter([i+0.5 for i in reorder_indices], 
                   [sim.order_amount/2] * len(reorder_indices),
                   color='green', marker='^', s=50, label='Reorder')
    
    plt.title('Inventory Level Simulation')
    plt.xlabel('Day')
    plt.ylabel('Inventory Level')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compute and visualize the stationary distribution
    try:
        stationary_distribution = sim.compute_stationary_distribution()
        
        plt.figure(figsize=(10, 6))
        states = sim.states
        plt.bar(states, stationary_distribution, color='skyblue')
        plt.title('Long-Term Inventory Level Distribution')
        plt.xlabel('Inventory Level')
        plt.ylabel('Probability')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate expected long-term inventory level
        expected_inventory = sum(state * prob for state, prob in zip(states, stationary_distribution))
        print(f"Expected long-term inventory level: {expected_inventory:.2f}")
        
        # Calculate long-term stockout probability
        stockout_prob = stationary_distribution[0]
        print(f"Long-term stockout probability: {stockout_prob:.2%}")
    
    except ValueError as e:
        print(f"Could not compute stationary distribution: {e}")
    
    # Visualize the transition matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim.transition_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=sim.states,
        yticklabels=sim.states
    )
    plt.title('Inventory Transition Probabilities')
    plt.xlabel('Next Day\'s Inventory')
    plt.ylabel('Current Day\'s Inventory')
    plt.tight_layout()
    plt.show()
    
    # Run a sensitivity analysis for different order amounts
    order_amounts = [3, 5, 7, 9]
    results = []
    
    for order_amount in order_amounts:
        test_sim = create_inventory_model(
            demand_probs=[0.2, 0.3, 0.3, 0.15, 0.05],
            max_inventory=10,
            order_amount=order_amount,
            days=1000  # Longer simulation for better statistics
        )
        test_sim.run_simulation()
        
        inventory_seq = test_sim.get_state_names()
        avg_inventory = sum(inventory_seq) / len(inventory_seq)
        stockout_rate = inventory_seq.count(0) / len(inventory_seq)
        
        results.append({
            'order_amount': order_amount,
            'avg_inventory': avg_inventory,
            'stockout_rate': stockout_rate
        })
    
    # Plot sensitivity analysis results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average inventory
    order_vals = [r['order_amount'] for r in results]
    avg_vals = [r['avg_inventory'] for r in results]
    stockout_vals = [r['stockout_rate'] for r in results]
    
    ax1.plot(order_vals, avg_vals, 'bo-', linewidth=2)
    ax1.set_title('Average Inventory Level')
    ax1.set_xlabel('Order Amount')
    ax1.set_ylabel('Average Inventory')
    ax1.grid(alpha=0.3)
    
    # Plot stockout rate
    ax2.plot(order_vals, stockout_vals, 'ro-', linewidth=2)
    ax2.set_title('Stockout Rate')
    ax2.set_xlabel('Order Amount')
    ax2.set_ylabel('Stockout Rate')
    ax2.set_ylim(0, max(stockout_vals) * 1.2)  # Add some margin
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sim


def create_custom_markov_chain():
    """Create and analyze a custom Markov chain simulation.
    
    This example demonstrates how to create a Markov chain from scratch,
    rather than using one of the predefined models.
    """
    print("\nCreating custom Markov chain simulation...")
    
    # Define a custom transition matrix for a simple game
    # States: "Start", "Level1", "Level2", "Win", "Lose"
    transition_matrix = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Start -> Level1
        [0.1, 0.2, 0.5, 0.1, 0.1],  # Level1 -> Start, Level1, Level2, Win, Lose
        [0.0, 0.3, 0.3, 0.3, 0.1],  # Level2 -> Level1, Level2, Win, Lose
        [0.0, 0.0, 0.0, 1.0, 0.0],  # Win is absorbing state
        [0.0, 0.0, 0.0, 0.0, 1.0]   # Lose is absorbing state
    ])
    
    # Define states
    states = ["Start", "Level1", "Level2", "Win", "Lose"]
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "MarkovChain",
        transition_matrix=transition_matrix,
        states=states,
        initial_state="Start",
        days=50
    )
    
    # Run multiple simulations to analyze outcomes
    num_simulations = 1000
    outcomes = {"Win": 0, "Lose": 0, "Incomplete": 0}
    game_lengths = []
    
    for _ in range(num_simulations):
        sim.reset()
        sim.run_simulation()
        
        # Get the game path
        game_path = sim.get_state_names()
        final_state = game_path[-1]
        
        # Record outcome
        if final_state == "Win":
            outcomes["Win"] += 1
        elif final_state == "Lose":
            outcomes["Lose"] += 1
        else:
            outcomes["Incomplete"] += 1
        
        # Record game length (excluding absorbing states at the end)
        if final_state in ["Win", "Lose"]:
            # Find when we first hit the absorbing state
            absorbing_index = game_path.index(final_state)
            game_lengths.append(absorbing_index)
    
    # Print statistics
    print("\nGame outcome statistics (from 1000 simulations):")
    total = sum(outcomes.values())
    for outcome, count in outcomes.items():
        print(f"{outcome}: {count} ({count/total:.1%})")
    
    if game_lengths:
        print(f"Average game length: {sum(game_lengths)/len(game_lengths):.1f} steps")
        print(f"Shortest game: {min(game_lengths)} steps")
        print(f"Longest game: {max(game_lengths)} steps")
    
    # Visualize the transition matrix as a colored network diagram
    plt.figure(figsize=(10, 8))
    
    # Create a circular layout for the states
    angles = np.linspace(0, 2*np.pi, len(states), endpoint=False)
    pos = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    
    # Draw nodes
    node_colors = ['#FFD700', '#87CEEB', '#98FB98', '#90EE90', '#FFA07A']
    for i, state in enumerate(states):
        plt.scatter(pos[i][0], pos[i][1], s=1000, color=node_colors[i], edgecolors='black', zorder=2)
        plt.text(pos[i][0], pos[i][1], state, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw edges
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = transition_matrix[i, j]
            if prob > 0:
                # Adjust the curvature for self-loops
                if i == j:
                    # Draw a loop
                    loop_radius = 0.2
                    center_x = pos[i][0] + loop_radius
                    center_y = pos[i][1]
                    circle = plt.Circle((center_x, center_y), loop_radius, fill=False, 
                                        linestyle='-', linewidth=1 + 5*prob, alpha=0.7)
                    plt.gca().add_patch(circle)
                    plt.text(center_x, center_y + loop_radius + 0.05, f"{prob:.1f}", 
                             ha='center', va='center', fontsize=10)
                else:
                    # Draw a straight line
                    plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 
                             'k-', alpha=0.5, linewidth=1 + 5*prob, zorder=1)
                    
                    # Add probability label at midpoint
                    mid_x = (pos[i][0] + pos[j][0]) / 2
                    mid_y = (pos[i][1] + pos[j][1]) / 2
                    plt.text(mid_x, mid_y, f"{prob:.1f}", ha='center', va='center', 
                             fontsize=10, fontweight='bold', zorder=3,
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.title('Game State Transition Network')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Plot distribution of game lengths
    if game_lengths:
        plt.figure(figsize=(10, 6))
        plt.hist(game_lengths, bins=range(min(game_lengths), max(game_lengths) + 2), 
                 color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=sum(game_lengths)/len(game_lengths), color='red', linestyle='--', 
                   label=f'Average: {sum(game_lengths)/len(game_lengths):.1f}')
        plt.title('Distribution of Game Lengths')
        plt.xlabel('Number of Steps')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return sim


# Run simulations
if __name__ == "__main__":
    # Run weather forecast simulation
    weather_sim = run_weather_simulation()
    
    # Run random walk simulation
    random_walk_sim = run_random_walk_simulation()
    
    # Run inventory management simulation
    inventory_sim = run_inventory_simulation()
    
    # Create and analyze a custom Markov chain
    custom_sim = create_custom_markov_chain()