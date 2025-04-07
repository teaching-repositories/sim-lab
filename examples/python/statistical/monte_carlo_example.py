"""
Example of using SimLab to create Monte Carlo simulations.

This example demonstrates:
1. Creating and running Monte Carlo simulations
2. Estimating Pi using random sampling
3. Analyzing option pricing with Monte Carlo methods
4. Estimating probabilities with confidence intervals
5. Analyzing convergence as sample size increases
"""

from sim_lab.core import SimulatorRegistry
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from typing import Dict, List, Tuple
from mpl_toolkits.mplot3d import Axes3D


def estimate_pi():
    """
    Monte Carlo simulation to estimate the value of Pi.
    
    This method uses the relationship between the area of a circle
    and the area of its enclosing square to estimate Pi.
    """
    # Define sample function - random point in a square
    def sample_function():
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        return (x, y)
    
    # Define evaluation function - check if point is in circle
    def evaluation_function(point):
        x, y = point
        # Return 4 if in circle (to scale to pi), 0 otherwise
        return 4.0 if x*x + y*y <= 1 else 0.0
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "MonteCarlo",
        sample_function=sample_function,
        evaluation_function=evaluation_function,
        num_samples=10000,
        days=20,
        confidence_interval=True,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    confidence_intervals = sim.get_confidence_intervals()
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), results, marker='o', linestyle='-', label='Estimated Pi')
    
    # Plot confidence intervals
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    plt.fill_between(range(1, len(results) + 1), lower_bounds, upper_bounds, alpha=0.3, label='95% Confidence Interval')
    
    # Plot actual Pi for reference
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Actual Pi')
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Estimated Value of Pi')
    plt.title('Monte Carlo Estimation of Pi')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Analyze convergence
    convergence = sim.get_convergence_analysis()
    
    plt.figure(figsize=(10, 6))
    plt.plot(convergence['sample_sizes'], convergence['means'], marker='o')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Actual Pi')
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated Value of Pi')
    plt.title('Convergence of Pi Estimation by Sample Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Visualize points in the final simulation step
    plt.figure(figsize=(8, 8))
    
    # Generate points for visualization (not the actual simulation points)
    np.random.seed(42)
    points = [sample_function() for _ in range(2000)]
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    colors = ['blue' if evaluation_function(p) > 0 else 'red' for p in points]
    
    # Plot points
    plt.scatter(x_values, y_values, c=colors, alpha=0.6, s=20)
    
    # Draw the circle and square
    circle = plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2)
    plt.gca().add_patch(circle)
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2)
    
    plt.axis('equal')
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title('Monte Carlo Estimation of Pi: Points Distribution')
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'pi_estimates': results,
        'confidence_intervals': confidence_intervals,
        'final_estimate': results[-1],
        'actual_pi': np.pi,
        'error': abs(results[-1] - np.pi) / np.pi * 100  # Percent error
    }


def option_pricing_simulation():
    """
    Monte Carlo simulation for European option pricing.
    
    This simulation models the Black-Scholes model for pricing a European call option.
    """
    # Define parameters
    S0 = 100.0       # Initial stock price
    K = 105.0        # Strike price
    r = 0.05         # Risk-free interest rate
    sigma = 0.2      # Volatility
    T = 1.0          # Time to maturity in years
    
    # Define sample function - generate a random stock price path
    def sample_function():
        # Generate a random standard normal number
        z = random.gauss(0.0, 1.0)
        
        # Use the Black-Scholes model to calculate the final stock price
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        return ST
    
    # Define evaluation function - calculate option payoff
    def evaluation_function(ST):
        # European call option payoff: max(ST - K, 0)
        payoff = max(ST - K, 0)
        # Discount to present value
        present_value = payoff * np.exp(-r * T)
        return present_value
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "MonteCarlo",
        sample_function=sample_function,
        evaluation_function=evaluation_function,
        num_samples=10000,
        days=30,
        confidence_interval=True,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    confidence_intervals = sim.get_confidence_intervals()
    
    # Calculate the Black-Scholes analytical solution for comparison
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    
    def norm_cdf(x):
        return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0
    
    analytical_price = black_scholes_call(S0, K, T, r, sigma)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), results, marker='o', linestyle='-', 
             label='Monte Carlo Price')
    
    # Plot confidence intervals
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    plt.fill_between(range(1, len(results) + 1), lower_bounds, upper_bounds, alpha=0.3, 
                     label='95% Confidence Interval')
    
    # Plot analytical solution for reference
    plt.axhline(y=analytical_price, color='r', linestyle='--', 
                label='Black-Scholes Analytical Price')
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Simulation for European Call Option Pricing')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Display final price distribution
    plt.figure(figsize=(10, 6))
    final_payoffs = [evaluation_function(sample_function()) for _ in range(5000)]
    plt.hist(final_payoffs, bins=50, alpha=0.7, density=True)
    plt.axvline(x=results[-1], color='r', linestyle='--', 
                label=f'Monte Carlo Price: {results[-1]:.4f}')
    plt.axvline(x=analytical_price, color='g', linestyle='--', 
                label=f'Black-Scholes Price: {analytical_price:.4f}')
    plt.xlabel('Option Payoff')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Option Payoffs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'option_price_estimates': results,
        'confidence_intervals': confidence_intervals,
        'final_estimate': results[-1],
        'analytical_price': analytical_price,
        'error': abs(results[-1] - analytical_price) / analytical_price * 100  # Percent error
    }


def project_completion_simulation():
    """
    Monte Carlo simulation for project completion time estimation.
    
    This simulation models the completion time of a project with multiple tasks,
    each with uncertain durations.
    """
    # Define project tasks with optimistic, most likely, and pessimistic durations
    tasks = [
        {'name': 'Task A', 'optimistic': 2, 'most_likely': 4, 'pessimistic': 8},
        {'name': 'Task B', 'optimistic': 3, 'most_likely': 5, 'pessimistic': 10},
        {'name': 'Task C', 'optimistic': 4, 'most_likely': 7, 'pessimistic': 12},
        {'name': 'Task D', 'optimistic': 6, 'most_likely': 9, 'pessimistic': 15},
        {'name': 'Task E', 'optimistic': 3, 'most_likely': 6, 'pessimistic': 9}
    ]
    
    # Define sample function - generate random task durations
    def sample_function():
        task_durations = []
        for task in tasks:
            # Use PERT distribution (beta distribution with parameters from the three-point estimate)
            a = task['optimistic']
            m = task['most_likely']
            b = task['pessimistic']
            
            # Generate a beta random variable
            alpha = 1 + 4 * (m - a) / (b - a)
            beta = 1 + 4 * (b - m) / (b - a)
            
            # Handle case where alpha or beta are undefined
            if not (alpha > 0 and beta > 0 and b > a):
                duration = (a + 4*m + b) / 6  # Use PERT expected value
            else:
                # Generate from beta distribution and scale to [a, b]
                x = random.betavariate(alpha, beta)
                duration = a + x * (b - a)
            
            task_durations.append(duration)
        
        return task_durations
    
    # Define evaluation function - calculate project completion time
    def evaluation_function(task_durations):
        # In this simple model, tasks are sequential, so the completion time is the sum
        # In a real project, you'd use a critical path calculation
        return sum(task_durations)
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "MonteCarlo",
        sample_function=sample_function,
        evaluation_function=evaluation_function,
        num_samples=5000,
        days=10,
        confidence_interval=True,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    confidence_intervals = sim.get_confidence_intervals()
    statistics = sim.get_statistics()
    
    # Calculate the deterministic estimate (PERT expected value)
    deterministic_durations = [(t['optimistic'] + 4*t['most_likely'] + t['pessimistic'])/6 for t in tasks]
    deterministic_estimate = sum(deterministic_durations)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), results, marker='o', linestyle='-', 
             label='Monte Carlo Estimate')
    
    # Plot confidence intervals
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    plt.fill_between(range(1, len(results) + 1), lower_bounds, upper_bounds, alpha=0.3, 
                     label='95% Confidence Interval')
    
    # Plot deterministic estimate for reference
    plt.axhline(y=deterministic_estimate, color='r', linestyle='--', 
                label='Deterministic Estimate (PERT)')
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Project Completion Time')
    plt.title('Monte Carlo Simulation for Project Completion Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Generate histogram of completion times
    plt.figure(figsize=(12, 6))
    all_completions = [evaluation_function(sample_function()) for _ in range(10000)]
    
    plt.hist(all_completions, bins=30, alpha=0.7, density=True)
    plt.axvline(x=statistics['mean'], color='blue', linestyle='-', 
                label=f'Mean: {statistics["mean"]:.1f}')
    plt.axvline(x=statistics['percentiles']['50th'], color='green', linestyle='--',
                label=f'Median: {statistics["percentiles"]["50th"]:.1f}')
    plt.axvline(x=statistics['percentiles']['95th'], color='red', linestyle='--',
                label=f'95th Percentile: {statistics["percentiles"]["95th"]:.1f}')
    plt.axvline(x=deterministic_estimate, color='purple', linestyle='-.',
                label=f'PERT Estimate: {deterministic_estimate:.1f}')
    
    plt.xlabel('Project Completion Time')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Project Completion Times')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'completion_time_estimates': results,
        'confidence_intervals': confidence_intervals,
        'statistics': statistics,
        'deterministic_estimate': deterministic_estimate
    }


def portfolio_risk_simulation():
    """
    Monte Carlo simulation for portfolio risk analysis.
    
    This simulation models the returns of a portfolio with multiple assets,
    each with different expected returns, volatilities, and correlations.
    """
    # Define portfolio assets
    assets = [
        {'name': 'Stock A', 'expected_return': 0.08, 'volatility': 0.20},
        {'name': 'Stock B', 'expected_return': 0.12, 'volatility': 0.30},
        {'name': 'Bond', 'expected_return': 0.04, 'volatility': 0.05},
        {'name': 'Real Estate', 'expected_return': 0.09, 'volatility': 0.15}
    ]
    
    # Define correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.50, 0.10, 0.30],
        [0.50, 1.00, -0.05, 0.20],
        [0.10, -0.05, 1.00, 0.15],
        [0.30, 0.20, 0.15, 1.00]
    ])
    
    # Define portfolio weights
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    
    # Calculate covariance matrix
    volatilities = np.array([asset['volatility'] for asset in assets])
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Time horizon for simulation in years
    time_horizon = 5
    
    # Define sample function - generate random portfolio returns
    def sample_function():
        # Generate correlated returns using multivariate normal distribution
        annual_returns = np.random.multivariate_normal(
            mean=[asset['expected_return'] for asset in assets],
            cov=covariance_matrix,
            size=time_horizon
        )
        
        # Calculate portfolio value over time
        portfolio_value = 1.0  # Initial $1 investment
        for year_returns in annual_returns:
            # Portfolio return is the weighted sum of asset returns
            portfolio_return = np.sum(year_returns * weights)
            portfolio_value *= (1 + portfolio_return)
        
        return portfolio_value
    
    # Define evaluation function - calculate final portfolio value
    def evaluation_function(portfolio_value):
        return portfolio_value
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "MonteCarlo",
        sample_function=sample_function,
        evaluation_function=evaluation_function,
        num_samples=5000,
        days=20,
        confidence_interval=True,
        random_seed=42
    )
    
    # Run the simulation
    results = sim.run_simulation()
    confidence_intervals = sim.get_confidence_intervals()
    statistics = sim.get_statistics()
    
    # Calculate the expected portfolio value deterministically
    expected_annual_return = np.sum([weights[i] * assets[i]['expected_return'] for i in range(len(assets))])
    expected_final_value = (1 + expected_annual_return) ** time_horizon
    
    # Visualize convergence
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), results, marker='o', linestyle='-', 
             label='Monte Carlo Estimate')
    
    # Plot confidence intervals
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    plt.fill_between(range(1, len(results) + 1), lower_bounds, upper_bounds, alpha=0.3, 
                     label='95% Confidence Interval')
    
    # Plot deterministic estimate for reference
    plt.axhline(y=expected_final_value, color='r', linestyle='--', 
                label='Expected Value (no risk)')
    
    plt.xlabel('Simulation Step')
    plt.ylabel(f'Portfolio Value after {time_horizon} Years')
    plt.title('Monte Carlo Simulation for Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Generate histogram of portfolio values
    plt.figure(figsize=(12, 6))
    all_values = [sample_function() for _ in range(10000)]
    
    plt.hist(all_values, bins=50, alpha=0.7, density=True)
    plt.axvline(x=statistics['mean'], color='blue', linestyle='-', 
                label=f'Mean: ${statistics["mean"]:.2f}')
    plt.axvline(x=statistics['percentiles']['5th'], color='red', linestyle='--',
                label=f'5th Percentile (VaR 95%): ${statistics["percentiles"]["5th"]:.2f}')
    plt.axvline(x=statistics['percentiles']['1st'], color='purple', linestyle='--',
                label=f'1st Percentile (VaR 99%): ${statistics["percentiles"]["1st"]:.2f}')
    plt.axvline(x=expected_final_value, color='green', linestyle='-.',
                label=f'Expected Value: ${expected_final_value:.2f}')
    
    plt.xlabel(f'Portfolio Value after {time_horizon} Years')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Portfolio Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate Value at Risk (VaR) and Expected Shortfall (ES)
    var_95 = np.percentile(all_values, 5)
    var_99 = np.percentile(all_values, 1)
    
    es_95 = np.mean([v for v in all_values if v <= var_95])
    es_99 = np.mean([v for v in all_values if v <= var_99])
    
    # Return results
    return {
        'portfolio_value_estimates': results,
        'confidence_intervals': confidence_intervals,
        'statistics': statistics,
        'expected_final_value': expected_final_value,
        'var_95': var_95,
        'var_99': var_99,
        'es_95': es_95,
        'es_99': es_99
    }


# Run Monte Carlo simulations
if __name__ == "__main__":
    # 1. Estimate Pi using Monte Carlo
    print("1. Running Pi Estimation Simulation...")
    pi_results = estimate_pi()
    print(f"Final Pi estimate: {pi_results['final_estimate']:.6f}")
    print(f"Actual Pi value:   {pi_results['actual_pi']:.6f}")
    print(f"Percent error:     {pi_results['error']:.4f}%")
    print()
    
    # 2. Option Pricing Simulation
    print("2. Running Option Pricing Simulation...")
    option_results = option_pricing_simulation()
    print(f"Final option price estimate: ${option_results['final_estimate']:.4f}")
    print(f"Black-Scholes price:         ${option_results['analytical_price']:.4f}")
    print(f"Percent error:               {option_results['error']:.4f}%")
    print()
    
    # 3. Project Completion Time Simulation
    print("3. Running Project Completion Time Simulation...")
    project_results = project_completion_simulation()
    stats = project_results['statistics']
    print(f"Expected completion time:    {stats['mean']:.1f} days")
    print(f"Median completion time:      {stats['percentiles']['50th']:.1f} days")
    print(f"95th percentile completion:  {stats['percentiles']['95th']:.1f} days")
    print(f"PERT estimate:               {project_results['deterministic_estimate']:.1f} days")
    print()
    
    # 4. Portfolio Risk Simulation
    print("4. Running Portfolio Risk Simulation...")
    portfolio_results = portfolio_risk_simulation()
    stats = portfolio_results['statistics']
    print(f"Expected portfolio value:    ${portfolio_results['expected_final_value']:.2f}")
    print(f"Mean portfolio value:        ${stats['mean']:.2f}")
    print(f"Value at Risk (95%):         ${portfolio_results['var_95']:.2f}")
    print(f"Expected Shortfall (95%):    ${portfolio_results['es_95']:.2f}")
    print(f"Probability of loss:         {len([v for v in stats['all_results'] if v < 1]) / len(stats['all_results']):.2%}")
    print()