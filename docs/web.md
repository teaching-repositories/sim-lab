# Web Interface

SimNexus provides a web interface built with FastHTML for running simulations through a browser.

## Installation

Make sure you have installed SimNexus with the web dependencies:

```bash
pip install simnexus[web]
```

## Running the Web Server

Start the web server with:

```bash
simnexus ui web
```

By default, the server runs on http://localhost:8000.

For custom host and port:

```bash
simnexus ui web --host 0.0.0.0 --port 5000
```

## Web Interface Features

Navigate to http://localhost:8000 in your browser to access the web interface, which provides:

- Interactive forms for configuring simulations
- Visual charts of simulation results (powered by Chart.js)
- Statistical analysis of simulation results
- Options to download simulation data as CSV

## Simulation Types

### Stock Market Simulation 

Configure and run stock price simulations with parameters:

- Starting Price
- Days to simulate
- Volatility
- Drift (daily price trend)
- Event Day (optional)
- Event Impact

The simulation visualizes market events as markers on the chart, making it easy to see the impact of major events.

### Resource Fluctuations Simulation

Model resource price changes with parameters:

- Starting Price
- Days to simulate
- Volatility
- Drift
- Supply Disruption Day (optional)
- Disruption Severity

### Product Popularity Simulation

Coming soon!

## Behind the Scenes - FastHTML

The SimNexus web interface is built using FastHTML, which combines:

- Starlette for ASGI web server
- HTMX for interactive browser experiences
- FastTags (FT) for dynamic HTML generation

FastHTML allows SimNexus to create a modern, interactive web interface without the complexity of a full JavaScript framework.

## Deployment

For production deployment, we recommend using Uvicorn behind a reverse proxy like Nginx:

```bash
uvicorn simnexus.web.app:create_app --host 0.0.0.0 --port 8000
```

Or with Docker (see the project repository for the Dockerfile).