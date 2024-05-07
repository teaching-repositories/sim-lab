# SimNexus: Business Simulation Toolkit

This Python package provides a set of classes for simulating various
business-related scenarios. It is designed for educational use, allowing
students to experiment with modeling, analysis, and decision-making in different
contexts.

## Installation

To install SimNexus, you can use pip directly from GitHub:

```bash
pip install git+https://github.com/teaching-repositories/simnexus.git
```

## Available Simulations

- **Stock Market Simulation:** Simulate stock price fluctuations, incorporate
  technical indicators, and develop simple trading strategies.
- **Resource Fluctuations Simulation:** Model changes in the price of a
  resource, analyze supply/demand dynamics, and implement hedging strategies.
- **Product Popularity Simulation:** Simulate the rise and fall of product
  demand, investigate virality factors, and examine different marketing
  strategies.

## Basic Usage Example

Here is an example of how to use the Resource Fluctuations Simulation:

```python
from simnexus import ResourceFluctuationsSimulation

# Create a resource simulation
sim = ResourceFluctuationsSimulation(start_price=100, days=365, volatility=0.05, drift=0.01,
                         supply_disruption_day=180, disruption_severity=0.2)

# Run the simulation and get results
prices = sim.run_simulation()

# Visualize or analyze the results here
```

## Project Goals (For Students)

- Gain familiarity with using simulations to model dynamic systems.
- Understand the impact of different parameters on simulation outcomes.
- Develop data analysis and visualization skills in the context of business problems.
- Practice translating business concepts into simulation parameters.

## Contributing

We welcome contributions to the SimNexus project! If you have suggestions or improvements, feel free to fork the repository and submit a pull request. Please ensure your code adheres to the existing style, and include tests for new features or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
