# Getting Started with Simulacra

Welcome to the getting started guide for Simulacra, a package designed for simulating various phenomena such as product popularity, stock market behaviors, and resource fluctuations. This guide will walk you through the steps of installing the package and running your first simulation.

## Installation

To use Simulacra, you need to have Python installed on your machine. If you do not have Python installed, you can download and install it from [python.org](https://www.python.org/downloads/).

### Install from GitHub

Simulacra can be installed directly from its GitHub repository using pip. Open your command line interface (CLI) and run the following command:

```bash
pip install git+https://github.com/teaching-repositories/simulacra.git -q
```

This command will fetch the latest version of Simulacra from the GitHub repository and install it along with its dependencies.

## Verify Installation

To ensure that Simulacra was installed correctly, try running the following command:

```python
python -c "import simulacra; print(simulacra.__version__)"
```

This command should print the version number of the Simulacra package if it has been installed successfully.

## Running Your First Simulation

Once Simulacra is installed, you can start simulating right away. Here’s a quick example on how to simulate product popularity:

```python
from simulacra import ProductPopularitySimulation

# Create a simulation instance
sim = ProductPopularitySimulation(
    start_demand=100,
    days=365,
    growth_rate=0.01,
    marketing_impact=0.05,
    promotion_day=100,
    promotion_effectiveness=0.3,
    random_seed=42
)

# Run the simulation
results = sim.run_simulation()

# Print the results
print(results)
```

This example sets up a year-long simulation of product demand, including a promotion day with specific effectiveness.

## Next Steps

- Explore the detailed API documentation for more features and other simulation types.
- Check out examples and tutorials in the documentation to get more familiar with what you can achieve with Simulacra.

## Getting Help

If you encounter any issues or have questions, please refer to the [Contact](contact.md) page for information on how to get in touch.

Thank you for using Simulacra, and happy simulating!
