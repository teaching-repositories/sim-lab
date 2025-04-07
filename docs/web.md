# Web Interface

SimLab provides a web interface built with FastHTML for running simulations through a browser.

## Installation

Make sure you have installed SimLab with the web dependencies:

```bash
pip install sim-lab[web]
```

## Running the Web Server

Start the web server with:

```bash
simlab ui web
```

By default, the server runs on http://localhost:8000.

For custom host and port:

```bash
simlab ui web --host 0.0.0.0 --port 5000
```

## Web Interface Features

Navigate to http://localhost:8000 in your browser to access the web interface, which provides:

- Interactive forms for configuring simulations
- Visual charts of simulation results (using Matplotlib)
- Statistical analysis of simulation results
- Options to download simulation data as CSV

## Simulation Types

### Stock Market Simulation 

Configure and run stock price simulations with parameters:

- Starting Price
- Days to simulate
- Volatility
- Drift (daily price trend)
- Market Event (optional)
  - Event Day
  - Event Impact

The simulation visualizes market events as vertical lines on the chart, making it easy to see the impact of major events.

The interface also provides:
- Key statistics (starting price, final price, overall change)
- Extremes (maximum and minimum prices with their corresponding days)
- Download options for the simulation data

### Resource Fluctuations Simulation

Model resource price changes with parameters:

- Starting Price
- Days to simulate
- Volatility
- Drift
- Supply Disruption Day (optional)
- Disruption Severity

### Product Popularity Simulation

Model how products gain market share through marketing and word-of-mouth effects with parameters:

- Initial Popularity
- Virality Factor
- Marketing Effectiveness
- Days to simulate

## Behind the Scenes - FastHTML

The SimLab web interface is built using FastHTML, which combines:

- Starlette for ASGI web server
- HTMX for interactive browser experiences without writing JavaScript
- Server-side rendering for a fast, SEO-friendly experience

FastHTML allows SimLab to create a modern, interactive web interface without the complexity of a full JavaScript framework.

### Key FastHTML Features Used

- HTMX for AJAX requests without writing JavaScript
- Server-side rendering of charts with Matplotlib
- Form validation and processing
- Interactive UI elements

## Deployment

For production deployment, we recommend using Uvicorn behind a reverse proxy like Nginx:

```bash
uvicorn sim_lab.web.app:create_app --host 0.0.0.0 --port 8000
```

For a more robust setup, use Gunicorn as a process manager:

```bash
gunicorn sim_lab.web.app:create_app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

You can also deploy the web interface using Docker:

```bash
# Build the Docker image
docker build -t simlab-web -f Dockerfile.web .

# Run the container
docker run -p 8000:8000 simlab-web
```

## Development

To contribute to the web interface, follow these steps:

1. Install development dependencies: `pip install sim-lab[web,dev]`
2. Run the development server: `simlab ui web --reload`
3. Make changes to the templates in `src/sim_lab/web/templates/`
4. Add new routes in `src/sim_lab/web/app.py`

The web interface uses:
- Bootstrap 5 for styling
- HTMX for interactivity
- Matplotlib for chart generation