# SimNexus: Business Simulation Toolkit

This Python package provides a set of classes for simulating various
business-related scenarios. It is designed for educational use, allowing
students to experiment with modeling, analysis, and decision-making in different
contexts.

## Installation

To install SimNexus with all features:

```bash
pip install simnexus
```

For specific interfaces only:

```bash
# CLI only
pip install simnexus[cli]

# Web interface
pip install simnexus[web]
```

## Features

- Multiple simulation types for business scenarios
- Command-line interface (CLI)
- Terminal user interface (TUI)
- Web interface with REST API
- Visualization tools
- Data import/export

## Available Simulations

- **Stock Market Simulation:** Simulate stock price fluctuations, incorporate
  technical indicators, and develop simple trading strategies.
- **Resource Fluctuations Simulation:** Model changes in the price of a
  resource, analyze supply/demand dynamics, and implement hedging strategies.
- **Product Popularity Simulation:** Simulate the rise and fall of product
  demand, investigate virality factors, and examine different marketing
  strategies.

## Usage

### Python API

```python
from simnexus import ResourceFluctuationsSimulation

# Create a resource simulation
sim = ResourceFluctuationsSimulation(
    start_price=100, 
    days=365, 
    volatility=0.05, 
    drift=0.01,
    supply_disruption_day=180, 
    disruption_severity=0.2
)

# Run the simulation and get results
prices = sim.run_simulation()

# Visualize the results
from simnexus.viz import plot_time_series
plot_time_series(
    data=prices,
    title="Resource Price Fluctuations",
    xlabel="Days",
    ylabel="Price ($)",
    events={180: "Supply Disruption"}
)
```

### Command Line Interface

```bash
# Run a stock market simulation
simnexus stock run --start-price 100 --days 365 --volatility 0.02 --drift 0.001 --output prices.csv

# Get help for all commands
simnexus --help
```

### Terminal UI

```bash
# Launch the interactive terminal UI
simnexus-tui
```

### Web Interface

```bash
# Start the web server
simnexus-web

# Then visit http://localhost:8000 in your browser
```

## Project Goals (For Students)

- Gain familiarity with using simulations to model dynamic systems
- Understand the impact of different parameters on simulation outcomes
- Develop data analysis and visualization skills in the context of business problems
- Practice translating business concepts into simulation parameters

## Documentation

For more detailed information about using SimNexus, see the [documentation](https://yourproject.readthedocs.io/).

## Development

SimNexus uses modern Python development tools:

- [uv](https://github.com/astral-sh/uv) for dependency management
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [pytest](https://docs.pytest.org/) for testing
- [MkDocs](https://www.mkdocs.org/) for documentation

To setup a development environment:

```bash
# Clone the repository
git clone https://github.com/teaching-repositories/simnexus.git
cd simnexus

# Run the setup script
./scripts/setup_dev.sh

# Or manually
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .[dev]
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for more details.

## Contributing

We welcome contributions to the SimNexus project! See the [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
