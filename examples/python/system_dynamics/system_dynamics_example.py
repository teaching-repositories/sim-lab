"""
Example of using SimLab to create system dynamics models.

This example demonstrates:
1. Creating system dynamics models with stocks, flows, and auxiliaries
2. Running predefined system dynamics models
3. Creating custom system dynamics models
4. Visualizing system behavior over time
"""

from sim_lab.core import SimulatorRegistry, Stock, Flow, Auxiliary
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any


def run_population_model(
    initial_population=1000,
    birth_rate=0.03,
    death_rate=0.01,
    days=100
):
    """Run a simple population growth model.
    
    Args:
        initial_population: Initial population size
        birth_rate: Rate of population births (per capita)
        death_rate: Rate of population deaths (per capita)
        days: Number of days to simulate
        
    Returns:
        The completed simulation
    """
    print(f"Running population model with birth rate {birth_rate} and death rate {death_rate}...")
    
    # Use the predefined model creator
    sim = SimulatorRegistry.create(
        "SystemDynamics",
        stocks={
            'population': Stock('population', initial_population)
        },
        flows={
            'flow_from_births_to_population': Flow(
                'flow_from_births_to_population',
                lambda state, time: state['population'] * birth_rate
            ),
            'flow_from_population_to_deaths': Flow(
                'flow_from_population_to_deaths',
                lambda state, time: state['population'] * death_rate
            )
        },
        days=days,
        dt=0.1
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Return the simulation for further analysis
    return sim, results


def run_predator_prey_model(
    initial_prey=100,
    initial_predators=20,
    prey_growth_rate=0.1,
    predation_rate=0.01,
    predator_death_rate=0.05,
    predator_growth_rate=0.005,
    days=100
):
    """Run a predator-prey system dynamics model.
    
    Args:
        initial_prey: Initial prey population
        initial_predators: Initial predator population
        prey_growth_rate: Rate at which prey reproduce
        predation_rate: Rate at which predators consume prey
        predator_death_rate: Rate at which predators die
        predator_growth_rate: Rate at which predators convert prey to new predators
        days: Number of days to simulate
        
    Returns:
        The completed simulation
    """
    print("Running predator-prey system dynamics model...")
    
    # Use the predefined model creator
    sim = SimulatorRegistry.create(
        "SystemDynamics",
        stocks={
            'prey': Stock('prey', initial_prey),
            'predators': Stock('predators', initial_predators)
        },
        flows={
            'flow_from_growth_to_prey': Flow(
                'flow_from_growth_to_prey',
                lambda state, time: state['prey'] * prey_growth_rate
            ),
            'flow_from_prey_to_predation': Flow(
                'flow_from_prey_to_predation',
                lambda state, time: state['prey'] * state['predators'] * predation_rate
            ),
            'flow_from_predation_to_predators': Flow(
                'flow_from_predation_to_predators',
                lambda state, time: state['prey'] * state['predators'] * predator_growth_rate
            ),
            'flow_from_predators_to_death': Flow(
                'flow_from_predators_to_death',
                lambda state, time: state['predators'] * predator_death_rate
            )
        },
        days=days,
        dt=0.01
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Return the simulation for further analysis
    return sim, results


def run_sir_epidemic_model(
    population=10000,
    initial_infected=10,
    initial_recovered=0,
    transmission_rate=0.3,
    recovery_rate=0.1,
    days=100
):
    """Run an SIR epidemic model.
    
    Args:
        population: Total population size
        initial_infected: Initial number of infected individuals
        initial_recovered: Initial number of recovered individuals
        transmission_rate: Rate at which the disease spreads
        recovery_rate: Rate at which infected individuals recover
        days: Number of days to simulate
        
    Returns:
        The completed simulation
    """
    print("Running SIR epidemic model...")
    
    # Calculate initial susceptible population
    initial_susceptible = population - initial_infected - initial_recovered
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "SystemDynamics",
        stocks={
            'susceptible': Stock('susceptible', initial_susceptible),
            'infected': Stock('infected', initial_infected),
            'recovered': Stock('recovered', initial_recovered)
        },
        flows={
            'flow_from_susceptible_to_infected': Flow(
                'flow_from_susceptible_to_infected',
                lambda state, time: transmission_rate * state['susceptible'] * state['infected'] / population
            ),
            'flow_from_infected_to_recovered': Flow(
                'flow_from_infected_to_recovered',
                lambda state, time: recovery_rate * state['infected']
            )
        },
        days=days,
        dt=0.1
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Return the simulation for further analysis
    return sim, results


def run_custom_business_model(
    initial_customers=1000,
    initial_potential_customers=10000,
    word_of_mouth_effectiveness=0.01,
    advertising_effectiveness=0.03,
    customer_loss_fraction=0.02,
    days=200
):
    """Run a custom business growth system dynamics model.
    
    This model includes:
    - Potential customers who can become customers
    - Word-of-mouth effects
    - Advertising effects
    - Customer loss/churn
    
    Args:
        initial_customers: Initial customer base
        initial_potential_customers: Initial potential customers
        word_of_mouth_effectiveness: Effectiveness of word of mouth
        advertising_effectiveness: Effectiveness of advertising
        customer_loss_fraction: Fraction of customers lost per day
        days: Number of days to simulate
        
    Returns:
        The completed simulation
    """
    print("Running custom business growth model...")
    
    # Define stocks
    stocks = {
        'potential_customers': Stock('potential_customers', initial_potential_customers),
        'customers': Stock('customers', initial_customers)
    }
    
    # Define flows
    def word_of_mouth_adoption(state, time):
        """Rate at which potential customers become customers due to word of mouth."""
        # Adoption rate is proportional to current customers and potential customers
        return word_of_mouth_effectiveness * state['customers'] * state['potential_customers']
    
    def advertising_adoption(state, time):
        """Rate at which potential customers become customers due to advertising."""
        # Advertising reaches a fraction of potential customers
        return advertising_effectiveness * state['potential_customers']
    
    def customer_loss(state, time):
        """Rate at which customers are lost."""
        # A fraction of customers leave each day
        return customer_loss_fraction * state['customers']
    
    flows = {
        'flow_from_potential_customers_to_customers_wom': Flow(
            'flow_from_potential_customers_to_customers_wom', 
            word_of_mouth_adoption
        ),
        'flow_from_potential_customers_to_customers_ad': Flow(
            'flow_from_potential_customers_to_customers_ad', 
            advertising_adoption
        ),
        'flow_from_customers_to_lost': Flow(
            'flow_from_customers_to_lost', 
            customer_loss
        )
    }
    
    # Define auxiliary variables
    def total_adoption_rate(state, flow_rates, time):
        """Calculate the total adoption rate from all sources."""
        return (flow_rates['flow_from_potential_customers_to_customers_wom'] + 
                flow_rates['flow_from_potential_customers_to_customers_ad'])
    
    def market_saturation(state, flow_rates, time):
        """Calculate the market saturation percentage."""
        total_market = state['customers'] + state['potential_customers']
        return state['customers'] / total_market if total_market > 0 else 0
    
    auxiliaries = {
        'total_adoption_rate': Auxiliary('total_adoption_rate', total_adoption_rate),
        'market_saturation': Auxiliary('market_saturation', market_saturation)
    }
    
    # Create the simulation
    sim = SimulatorRegistry.create(
        "SystemDynamics",
        stocks=stocks,
        flows=flows,
        auxiliaries=auxiliaries,
        days=days,
        dt=0.2
    )
    
    # Run the simulation
    results = sim.run_simulation()
    
    # Return the simulation for further analysis
    return sim, results


def plot_system_dynamics_results(sim, results, title="System Dynamics Results"):
    """Visualize the results of a system dynamics simulation.
    
    Args:
        sim: The simulation object
        results: The simulation results
        title: Title for the plot
    """
    # Get time points for the x-axis
    time_points = np.linspace(0, sim.days, len(next(iter(results.values()))))
    
    # Separate stocks, flows, and auxiliaries
    stock_data = {k[6:]: v for k, v in results.items() if k.startswith('stock_')}
    flow_data = {k[5:]: v for k, v in results.items() if k.startswith('flow_')}
    aux_data = {k[4:]: v for k, v in results.items() if k.startswith('aux_')}
    
    # Plot stocks
    if stock_data:
        plt.figure(figsize=(12, 6))
        for name, values in stock_data.items():
            plt.plot(time_points, values, label=name)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{title} - Stocks')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Plot flows
    if flow_data:
        plt.figure(figsize=(12, 6))
        for name, values in flow_data.items():
            # Simplify flow names for readability
            simplified_name = name.replace('flow_from_', '').replace('_to_', '→')
            plt.plot(time_points, values, label=simplified_name)
        
        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.title(f'{title} - Flows')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Plot auxiliaries
    if aux_data:
        plt.figure(figsize=(12, 6))
        for name, values in aux_data.items():
            plt.plot(time_points, values, label=name)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{title} - Auxiliary Variables')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Run various system dynamics models
if __name__ == "__main__":
    # 1. Population growth model
    sim1, results1 = run_population_model(
        initial_population=1000, 
        birth_rate=0.03, 
        death_rate=0.01, 
        days=100
    )
    
    print("\nPopulation model results:")
    final_population = sim1.get_stock_history('population')[-1]
    print(f"Final population: {final_population:.2f}")
    plot_system_dynamics_results(sim1, results1, "Population Growth Model")
    
    # 2. Predator-prey model
    sim2, results2 = run_predator_prey_model(
        initial_prey=100,
        initial_predators=20,
        prey_growth_rate=0.1,
        predation_rate=0.01,
        predator_death_rate=0.05,
        predator_growth_rate=0.005,
        days=200
    )
    
    print("\nPredator-prey model results:")
    final_prey = sim2.get_stock_history('prey')[-1]
    final_predators = sim2.get_stock_history('predators')[-1]
    print(f"Final prey population: {final_prey:.2f}")
    print(f"Final predator population: {final_predators:.2f}")
    plot_system_dynamics_results(sim2, results2, "Predator-Prey Model")
    
    # 3. SIR epidemic model
    sim3, results3 = run_sir_epidemic_model(
        population=10000,
        initial_infected=10,
        initial_recovered=0,
        transmission_rate=0.3,
        recovery_rate=0.1,
        days=100
    )
    
    print("\nSIR epidemic model results:")
    final_susceptible = sim3.get_stock_history('susceptible')[-1]
    final_infected = sim3.get_stock_history('infected')[-1]
    final_recovered = sim3.get_stock_history('recovered')[-1]
    print(f"Final susceptible: {final_susceptible:.2f}")
    print(f"Final infected: {final_infected:.2f}")
    print(f"Final recovered: {final_recovered:.2f}")
    plot_system_dynamics_results(sim3, results3, "SIR Epidemic Model")
    
    # 4. Custom business growth model
    sim4, results4 = run_custom_business_model(
        initial_customers=1000,
        initial_potential_customers=10000,
        word_of_mouth_effectiveness=0.01,
        advertising_effectiveness=0.03,
        customer_loss_fraction=0.02,
        days=200
    )
    
    print("\nBusiness growth model results:")
    final_customers = sim4.get_stock_history('customers')[-1]
    final_potential = sim4.get_stock_history('potential_customers')[-1]
    final_saturation = sim4.get_auxiliary_history('market_saturation')[-1]
    print(f"Final customers: {final_customers:.2f}")
    print(f"Final potential customers: {final_potential:.2f}")
    print(f"Final market saturation: {final_saturation:.2%}")
    plot_system_dynamics_results(sim4, results4, "Business Growth Model")
    
    # Compare different scenarios for the business model
    plt.figure(figsize=(12, 6))
    
    # Baseline scenario
    plt.plot(
        np.linspace(0, 200, len(sim4.get_stock_history('customers'))),
        sim4.get_stock_history('customers'),
        label='Baseline'
    )
    
    # High word-of-mouth scenario
    sim5, _ = run_custom_business_model(
        initial_customers=1000,
        initial_potential_customers=10000,
        word_of_mouth_effectiveness=0.02,  # Double word-of-mouth
        advertising_effectiveness=0.03,
        customer_loss_fraction=0.02,
        days=200
    )
    plt.plot(
        np.linspace(0, 200, len(sim5.get_stock_history('customers'))),
        sim5.get_stock_history('customers'),
        label='High Word-of-Mouth'
    )
    
    # High advertising scenario
    sim6, _ = run_custom_business_model(
        initial_customers=1000,
        initial_potential_customers=10000,
        word_of_mouth_effectiveness=0.01,
        advertising_effectiveness=0.06,  # Double advertising
        customer_loss_fraction=0.02,
        days=200
    )
    plt.plot(
        np.linspace(0, 200, len(sim6.get_stock_history('customers'))),
        sim6.get_stock_history('customers'),
        label='High Advertising'
    )
    
    # Low churn scenario
    sim7, _ = run_custom_business_model(
        initial_customers=1000,
        initial_potential_customers=10000,
        word_of_mouth_effectiveness=0.01,
        advertising_effectiveness=0.03,
        customer_loss_fraction=0.01,  # Half churn rate
        days=200
    )
    plt.plot(
        np.linspace(0, 200, len(sim7.get_stock_history('customers'))),
        sim7.get_stock_history('customers'),
        label='Low Churn'
    )
    
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Customers')
    plt.title('Business Growth Model - Scenario Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()