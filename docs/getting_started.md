# Getting Started with SimNexus

Welcome to the getting started guide for SimNexus, a toolkit for simulating various business scenarios including stock market fluctuations, resource price dynamics, and product popularity. This guide will walk you through the installation options and different ways to use SimNexus.

## Installation Options

SimNexus offers flexible installation options depending on your needs:

### Basic Installation

For core simulation functionality only:

```bash
pip install simnexus
```

### Full Installation

For all features including CLI, web interface, and development tools:

```bash
pip install simnexus[dev]
```

### Component-Specific Installation

For specific interfaces:

```bash
# Command-line interface
pip install simnexus[cli]

# Web interface
pip install simnexus[web]
```

### Install from GitHub

For the latest development version:

```bash
pip install git+https://github.com/michael-borck/simnexus.git
```

## Verify Installation

To ensure that SimNexus was installed correctly:

```python
python -c "import simnexus; print(simnexus.__version__)"
```

Or using the CLI:

```bash
simnexus --version
```

## Usage Methods

SimNexus offers multiple ways to run simulations:

### 1. Python Library

Import SimNexus in your Python code:

```python
from simnexus.core.stock_market_simulation import StockMarketSimulation

# Create a simulation instance
sim = StockMarketSimulation(
    start_price=100,
    days=365,
    volatility=0.03,
    drift=0.001,
    event_day=180,
    event_impact=-0.2,
    random_seed=42
)

# Run the simulation
prices = sim.run_simulation()

# Use the results
print(f"Final price: ${prices[-1]:.2f}")
```

### 2. Command Line Interface

Run simulations directly from the command line:

```bash
# Stock market simulation
simnexus sim stock run --days 365 --event-day 180 --event-impact -0.2 --output results.csv

# Resource fluctuations simulation
simnexus sim resource run --volatility 0.05 --disruption-day 100 --disruption-severity 0.3
```

### 3. Web Interface

Launch the web interface for interactive simulation:

```bash
simnexus ui web
```

Then open your browser at http://localhost:8000.

### 4. Terminal UI

Launch the terminal user interface (TUI):

```bash
simnexus ui tui
```

## Example Resources

SimNexus includes comprehensive examples to help you get started:

- **Python Examples**: Basic scripts showing simulation usage
- **CLI Examples**: Shell scripts demonstrating command-line capabilities  
- **Jupyter Notebooks**: Interactive examples for data analysis

See the [examples directory](https://github.com/michael-borck/simnexus/tree/main/examples) in the repository.

## Next Steps

- Explore the [API documentation](api.md) for detailed information on simulation classes
- Try different interfaces: [CLI](cli.md), [TUI](tui.md), or [Web](web.md)
- Check out the [example code](https://github.com/michael-borck/simnexus/tree/main/examples) for practical applications

## Getting Help

If you encounter any issues or have questions:

- Check the [documentation](https://michael-borck.github.io/simnexus/)
- Run `simnexus --help` for CLI assistance
- Contact information is available on the [Contact](contact.md) page

Happy simulating with SimNexus\!
