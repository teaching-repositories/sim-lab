# Command Line Interface

SimLab provides a command-line interface (CLI) for running simulations, launching interfaces, and managing the application.

## Installation

Make sure you have installed SimLab with the CLI dependencies:

```bash
pip install sim-lab[cli]
```

## Command Structure

The CLI is structured with a hierarchical command system:

```
simlab
├── sim              Commands for running simulations
│   ├── stock        Stock market simulation commands
│   │   └── run      Run a stock market simulation
│   ├── resource     Resource fluctuations simulation commands
│   │   └── run      Run a resource fluctuations simulation
│   └── product      Product popularity simulation commands
│       └── run      Run a product popularity simulation
├── ui               Launch user interfaces
│   ├── web          Launch the web interface
│   └── tui          Launch the terminal user interface
└── util             Utility commands
    └── info         Show information about SimLab
```

## Running Simulations

### Stock Market Simulation

```bash
# Run a basic stock market simulation
simlab sim stock run --start-price 100 --days 365 --volatility 0.02 --drift 0.001

# Include a market event
simlab sim stock run --start-price 150 --days 500 --volatility 0.03 --drift 0.002 --event-day 250 --event-impact -0.15

# Save the results to a specific file
simlab sim stock run --start-price 100 --days 365 --volatility 0.02 --drift 0.001 --output my_simulation.csv

# Visualize the results (requires matplotlib)
simlab sim stock run --days 365 --event-day 180 --event-impact -0.2 --viz
```

### Resource Fluctuations Simulation

```bash
# Run a basic resource simulation
simlab sim resource run --start-price 50 --days 365 --volatility 0.05 --drift 0.001

# Include a supply disruption
simlab sim resource run --start-price 75 --days 500 --volatility 0.04 --drift 0.002 --disruption-day 200 --disruption-severity 0.3

# Save the results to a specific file
simlab sim resource run --start-price 50 --days 365 --volatility 0.05 --drift 0.001 --output resource_prices.csv

# Visualize the results (requires matplotlib)
simlab sim resource run --disruption-day 180 --disruption-severity 0.3 --viz
```

### Product Popularity Simulation

```bash
# Run a basic product popularity simulation
simlab sim product run --initial-popularity 0.01 --virality 0.1 --marketing 0.05 --days 365

# Save the results to a specific file
simlab sim product run --initial-popularity 0.02 --virality 0.15 --marketing 0.08 --days 300 --output product_popularity.csv

# Visualize the results (requires matplotlib)
simlab sim product run --initial-popularity 0.05 --virality 0.2 --marketing 0.1 --days 200 --viz
```

## Running User Interfaces

### Web Interface

```bash
# Start the web interface with default settings
simlab ui web

# Specify host and port
simlab ui web --host 0.0.0.0 --port 5000
```

### Terminal User Interface

```bash
# Start the terminal user interface
simlab ui tui
```

## Utility Commands

```bash
# Show information about SimLab
simlab util info

# Show version
simlab --version
```

## Getting Help

You can get help for any command by using the `--help` flag:

```bash
# Get general help
simlab --help

# Get help for a command group
simlab sim --help

# Get help for a specific simulation type
simlab sim stock --help

# Get help for a specific command
simlab sim stock run --help
```