# SimLab Examples

This directory contains example code showing how to use SimLab in different contexts.

## Directory Structure

- `python/`: Python script examples for using SimLab as a library, organized by simulation category
- `cli/`: Command-line interface examples showing how to use the CLI tools
- `jupyter/`: Jupyter notebook examples for interactive data analysis

## Python Examples

The Python examples are organized by simulation categories to demonstrate how to use different types of simulations in SimLab:

### Basic Simulations
- `basic/stock_market_example.py`: Stock market price simulation
- `basic/resource_fluctuations_example.py`: Resource price fluctuation simulation

### Statistical Simulations
- `statistical/monte_carlo_example.py`: Monte Carlo simulations for estimating Pi, option pricing, etc.
- `statistical/markov_chain_example.py`: Markov chain simulations for weather forecasting, random walks, etc.

### Domain-Specific Simulations
- `domain_specific/epidemiological_example.py`: SIR model for epidemic spread
- `domain_specific/supply_chain_example.py`: Supply chain modeling with factories, distributors, and retailers
- `domain_specific/cellular_automaton_example.py`: Cellular automata including Conway's Game of Life

### System Dynamics Simulations
- `system_dynamics/system_dynamics_example.py`: Stock and flow models with feedback loops

### Network Simulations
- `network/information_diffusion_example.py`: Information spread through different network topologies

### Ecological Simulations
- `ecological/predator_prey_example.py`: Agent-based predator-prey dynamics in an ecosystem

### Discrete Event Simulations
- `discrete_event/bank_queue_example.py`: Bank queueing system with customer arrivals and service

### Miscellaneous
- `misc/simulator_registry_example.py`: Using the simulator registry to create simulations
- `misc/advanced_simulators_example.py`: Advanced usage of simulation classes
- `misc/advanced_ecosystem_simulators_example.py`: Advanced ecological simulations

To run any example:

```bash
cd python
python <category>/<example_file>.py

# For example:
python basic/stock_market_example.py
python statistical/monte_carlo_example.py
python domain_specific/epidemiological_example.py
```

## CLI Examples

The CLI examples show how to use SimLab from the command line:

- `run_simulations.sh`: Script demonstrating various CLI commands for different simulators

To run the CLI examples:

```bash
cd cli
chmod +x run_simulations.sh
./run_simulations.sh
```

Example CLI commands:

```bash
# Run a stock market simulation
simlab sim stock run --days 365 --volatility 0.03 --drift 0.001 --event-day 100 --event-impact -0.2

# Run a resource fluctuations simulation
simlab sim resource run --days 250 --volatility 0.015 --drift 0.0003 --disruption-day 100 --disruption-severity 0.3

# Run a product popularity simulation
simlab sim product run --initial-popularity 0.02 --virality 0.12 --marketing 0.05 --days 200
```

## Jupyter Examples

The Jupyter notebook examples demonstrate more complex scenarios and provide interactive data analysis:

- `simulation_examples.ipynb`: Comprehensive notebook with multiple simulations and analysis

To run the Jupyter examples:

```bash
cd jupyter
jupyter notebook simulation_examples.ipynb
```

## Web Interface

SimLab also provides a web interface for running simulations in a browser. To start the web interface:

```bash
simlab ui web
```

Then open your browser to http://localhost:8000 to access the web interface.

## Requirements

These examples require SimLab to be installed with the appropriate dependencies:

```bash
# For basic Python examples
pip install sim-lab

# For CLI examples
pip install sim-lab[cli]

# For Jupyter examples
pip install sim-lab[dev] pandas jupyter

# For web interface
pip install sim-lab[web]
```

## Example Descriptions

### Stock Market Simulation
Simulates stock price movements using geometric Brownian motion, allowing for price drift, volatility, and market events.

### Resource Fluctuations Simulation
Models resource price dynamics including supply disruptions and cyclical patterns.

### Monte Carlo Simulation
Demonstrates probabilistic simulations for estimating values, risk analysis, and option pricing.

### Markov Chain Simulation
Shows state-based simulations for weather forecasting, random walks, and inventory management.

### Cellular Automaton Simulation
Implements grid-based models where cells evolve based on rules and neighbor states.

### Agent-Based Simulation
Illustrates a predator-prey ecosystem where agent behaviors emerge complex system dynamics.

### Discrete Event Simulation
Models a bank queueing system with event scheduling based on arrival and service times.

### System Dynamics Simulation
Shows how to build models with stocks, flows, and feedback loops for business and ecosystem analysis.

### Network Simulation
Demonstrates information diffusion across different network structures.

### Supply Chain Simulation
Models a supply chain with factories, distributors, retailers, and different ordering policies.