# Command Line Interface

SimNexus provides a command-line interface (CLI) for running simulations, launching interfaces, and managing the application.

## Installation

Make sure you have installed SimNexus with the CLI dependencies:

```bash
pip install simnexus[cli]
```

## Command Structure

The CLI is structured with a hierarchical command system:

```
simnexus
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
    └── info         Show information about SimNexus
```

## Running Simulations

### Stock Market Simulation

```bash
# Run a basic stock market simulation
simnexus sim stock run --start-price 100 --days 365 --volatility 0.02 --drift 0.001

# Include a market event
simnexus sim stock run --start-price 150 --days 500 --volatility 0.03 --drift 0.002 --event-day 250 --event-impact -0.15

# Save the results to a specific file
simnexus sim stock run --start-price 100 --days 365 --volatility 0.02 --drift 0.001 --output my_simulation.csv

# Visualize the results (requires matplotlib)
simnexus sim stock run --days 365 --event-day 180 --event-impact -0.2 --viz
```

### Resource Fluctuations Simulation

```bash
# Run a basic resource simulation
simnexus sim resource run --start-price 50 --days 365 --volatility 0.05 --drift 0.001

# Include a supply disruption
simnexus sim resource run --start-price 75 --days 500 --volatility 0.04 --drift 0.002 --disruption-day 200 --disruption-severity 0.3

# Save the results to a specific file
simnexus sim resource run --start-price 50 --days 365 --volatility 0.05 --drift 0.001 --output resource_prices.csv

# Visualize the results (requires matplotlib)
simnexus sim resource run --disruption-day 180 --disruption-severity 0.3 --viz
```

## Running User Interfaces

### Web Interface

```bash
# Start the web interface with default settings
simnexus ui web

# Specify host and port
simnexus ui web --host 0.0.0.0 --port 5000

# Enable auto-reload for development
simnexus ui web --reload
```

### Terminal User Interface

```bash
# Start the terminal user interface
simnexus ui tui
```

## Utility Commands

```bash
# Show information about SimNexus
simnexus util info

# Show version
simnexus --version
```

## Getting Help

You can get help for any command by using the `--help` flag:

```bash
# Get general help
simnexus --help

# Get help for a command group
simnexus sim --help

# Get help for a specific simulation type
simnexus sim stock --help

# Get help for a specific command
simnexus sim stock run --help
```