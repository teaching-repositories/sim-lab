"""FastHTML web application for SimNexus."""

from fasthtml.common import *
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

from ..core import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)
from ..viz import plot_time_series

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create dataclasses for simulation parameters
@dataclass
class StockMarketParams:
    start_price: float = 100.0
    days: int = 365
    volatility: float = 0.02
    drift: float = 0.001
    event_day: Optional[int] = None
    event_impact: float = 0.0
    random_seed: Optional[int] = None

@dataclass
class ResourceFluctuationsParams:
    start_price: float = 100.0
    days: int = 365
    volatility: float = 0.03
    drift: float = 0.001
    supply_disruption_day: Optional[int] = None
    disruption_severity: float = 0.0
    random_seed: Optional[int] = None

# Create the FastHTML app
app, rt = fast_app(
    debug=True,
    hdrs=(
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        Style("""
            .simulation-chart {
                max-height: 400px;
                margin: 20px 0;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .results-area {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
    )
)

# Home page
@rt("/")
def get():
    """Return the home page."""
    content = Main(
        H1("SimNexus Simulations"),
        P("Welcome to SimNexus, a toolkit for running business simulations."),
        
        Div(
            H2("Available Simulations"),
            Div(
                Card(
                    H3("Stock Market Simulation"),
                    P("Simulate stock price fluctuations with various parameters."),
                    A("Run Simulation", href="/stock", cls="button"),
                    cls="stock-card"
                ),
                Card(
                    H3("Resource Fluctuations"),
                    P("Model changes in resource prices with supply disruptions."),
                    A("Run Simulation", href="/resource", cls="button"),
                    cls="resource-card"
                ),
                Card(
                    H3("Product Popularity"),
                    P("Simulate product demand patterns over time."),
                    A("Run Simulation", href="/product", cls="button"),
                    cls="product-card"
                ),
                cls="grid"
            )
        ),
        cls="container"
    )
    
    return Titled("SimNexus - Business Simulation Toolkit", content)

# Stock market simulation page
@rt("/stock")
def get():
    """Display the stock market simulation form."""
    form = Form(
        H2("Stock Market Simulation"),
        Div(
            Div(
                Label("Starting Price:"),
                Input(id="start_price", name="start_price", type="number", value="100.0", step="0.01"),
                cls="form-group"
            ),
            Div(
                Label("Days:"),
                Input(id="days", name="days", type="number", value="365", min="1"),
                cls="form-group"
            ),
            Div(
                Label("Volatility:"),
                Input(id="volatility", name="volatility", type="number", value="0.02", step="0.01", min="0"),
                cls="form-group"
            ),
            Div(
                Label("Drift:"),
                Input(id="drift", name="drift", type="number", value="0.001", step="0.001"),
                cls="form-group"
            ),
            Div(
                Label("Event Day (optional):"),
                Input(id="event_day", name="event_day", type="number", min="1"),
                cls="form-group"
            ),
            Div(
                Label("Event Impact:"),
                Input(id="event_impact", name="event_impact", type="number", value="0.0", step="0.01"),
                cls="form-group"
            ),
            Div(
                Label("Random Seed (optional):"),
                Input(id="random_seed", name="random_seed", type="number"),
                cls="form-group"
            ),
            Button("Run Simulation", type="submit"),
            
            # Results area that will be updated via HTMX
            Div(id="results-area", cls="results-area"),
            
            hx_post="/api/stock",
            hx_target="#results-area",
            hx_swap="innerHTML",
            hx_indicator="#spinner"
        ),
        Div(id="spinner", cls="htmx-indicator", "Loading..."),
        
        cls="container"
    )
    
    return Titled("Stock Market Simulation - SimNexus", form)

# Resource fluctuations simulation page
@rt("/resource")
def get():
    """Display the resource fluctuations simulation form."""
    form = Form(
        H2("Resource Fluctuations Simulation"),
        Div(
            Div(
                Label("Starting Price:"),
                Input(id="start_price", name="start_price", type="number", value="100.0", step="0.01"),
                cls="form-group"
            ),
            Div(
                Label("Days:"),
                Input(id="days", name="days", type="number", value="365", min="1"),
                cls="form-group"
            ),
            Div(
                Label("Volatility:"),
                Input(id="volatility", name="volatility", type="number", value="0.03", step="0.01", min="0"),
                cls="form-group"
            ),
            Div(
                Label("Drift:"),
                Input(id="drift", name="drift", type="number", value="0.001", step="0.001"),
                cls="form-group"
            ),
            Div(
                Label("Supply Disruption Day (optional):"),
                Input(id="supply_disruption_day", name="supply_disruption_day", type="number", min="1"),
                cls="form-group"
            ),
            Div(
                Label("Disruption Severity:"),
                Input(id="disruption_severity", name="disruption_severity", type="number", value="0.0", step="0.01"),
                cls="form-group"
            ),
            Div(
                Label("Random Seed (optional):"),
                Input(id="random_seed", name="random_seed", type="number"),
                cls="form-group"
            ),
            Button("Run Simulation", type="submit"),
            
            # Results area that will be updated via HTMX
            Div(id="results-area", cls="results-area"),
            
            hx_post="/api/resource",
            hx_target="#results-area",
            hx_swap="innerHTML",
            hx_indicator="#spinner"
        ),
        Div(id="spinner", cls="htmx-indicator", "Loading..."),
        
        cls="container"
    )
    
    return Titled("Resource Fluctuations Simulation - SimNexus", form)

# Product popularity simulation page (placeholder)
@rt("/product")
def get():
    """Display the product popularity simulation form."""
    return Titled(
        "Product Popularity Simulation - SimNexus", 
        Main(
            H2("Product Popularity Simulation"),
            P("This simulation is coming soon!"),
            A("Back to Home", href="/", cls="button"),
            cls="container"
        )
    )

# Stock market API endpoint
@rt("/api/stock")
def post(params: StockMarketParams):
    """Run a stock market simulation with the given parameters."""
    try:
        # Create and run the simulation
        sim = StockMarketSimulation(
            start_price=params.start_price,
            days=params.days,
            volatility=params.volatility,
            drift=params.drift,
            event_day=params.event_day,
            event_impact=params.event_impact,
            random_seed=params.random_seed
        )
        
        prices = sim.run_simulation()
        
        # Create unique chart ID
        chart_id = f"chart-{hash(tuple(prices)) % 10000}"
        
        # Create event data for chart if applicable
        events = {}
        if params.event_day is not None:
            events[params.event_day] = f"Market Event (Impact: {params.event_impact})"
        
        # Return HTML with visualization
        return create_result_view(
            "Stock Market Simulation Results", 
            prices, 
            params, 
            chart_id,
            events=events
        )
        
    except Exception as e:
        logger.error(f"Error running stock simulation: {e}")
        return Div(
            H3("Error"),
            P(f"An error occurred: {str(e)}"),
            cls="error"
        )

# Resource fluctuations API endpoint
@rt("/api/resource")
def post(params: ResourceFluctuationsParams):
    """Run a resource fluctuations simulation with the given parameters."""
    try:
        # Create and run the simulation
        sim = ResourceFluctuationsSimulation(
            start_price=params.start_price,
            days=params.days,
            volatility=params.volatility,
            drift=params.drift,
            supply_disruption_day=params.supply_disruption_day,
            disruption_severity=params.disruption_severity,
            random_seed=params.random_seed
        )
        
        prices = sim.run_simulation()
        
        # Create unique chart ID
        chart_id = f"chart-{hash(tuple(prices)) % 10000}"
        
        # Create event data for chart if applicable
        events = {}
        if params.supply_disruption_day is not None:
            events[params.supply_disruption_day] = f"Supply Disruption (Severity: {params.disruption_severity})"
        
        # Return HTML with visualization
        return create_result_view(
            "Resource Fluctuations Simulation Results", 
            prices, 
            params, 
            chart_id,
            events=events
        )
        
    except Exception as e:
        logger.error(f"Error running resource simulation: {e}")
        return Div(
            H3("Error"),
            P(f"An error occurred: {str(e)}"),
            cls="error"
        )

def create_result_view(title, data, params, chart_id, events=None):
    """Create a results view with a chart and data table."""
    # Create labels for the chart (days)
    labels = list(range(len(data)))
    
    # Mark events if any
    event_marks = ""
    if events:
        event_points = []
        for day, label in events.items():
            if 0 <= day < len(data):
                event_points.append({
                    'x': day,
                    'y': data[day],
                    'label': label
                })
        
        if event_points:
            event_marks = f"""
            annotation: {{
                annotations: {json.dumps(event_points)}.map(point => ({{
                    type: 'point',
                    xValue: point.x,
                    yValue: point.y,
                    backgroundColor: 'rgba(255, 0, 0, 0.5)',
                    borderColor: 'rgb(255, 0, 0)',
                    borderWidth: 2,
                    radius: 5,
                    label: {{
                        content: point.label,
                        enabled: true,
                        position: 'top'
                    }}
                }}))
            }},
            """
    
    # Create the chart
    chart_js = Script(f"""
    document.addEventListener('DOMContentLoaded', function() {{
        const ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: '{title}',
                    data: {json.dumps(data)},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Days'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Value'
                        }}
                    }}
                }},
                {event_marks}
                plugins: {{
                    title: {{
                        display: true,
                        text: '{title}'
                    }}
                }}
            }}
        }});
    }});
    """)

    # Create the parameters table
    param_rows = []
    for key, value in params.__dict__.items():
        if value is not None:
            param_rows.append(Tr(Td(key.replace('_', ' ').title()), Td(str(value))))
    
    # Create statistics
    min_val = min(data)
    max_val = max(data)
    final_val = data[-1]
    start_val = data[0]
    change = ((final_val - start_val) / start_val) * 100
    
    return Div(
        H3(title),
        Div(
            Canvas(id=chart_id, cls="simulation-chart"),
            chart_js
        ),
        Div(
            H4("Statistics"),
            Table(
                Tr(Th("Metric"), Th("Value")),
                Tr(Td("Starting Value"), Td(f"{start_val:.2f}")),
                Tr(Td("Final Value"), Td(f"{final_val:.2f}")),
                Tr(Td("Change"), Td(f"{change:.2f}%")),
                Tr(Td("Minimum"), Td(f"{min_val:.2f}")),
                Tr(Td("Maximum"), Td(f"{max_val:.2f}"))
            )
        ),
        Div(
            H4("Parameters"),
            Table(*param_rows)
        ),
        Div(
            H4("Download Results"),
            Button(
                "Download CSV",
                onclick=f"""
                    const data = {json.dumps([[i, val] for i, val in enumerate(data)])};
                    const csv = 'Day,Value\\n' + data.map(row => row.join(',')).join('\\n');
                    const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'simulation_results.csv';
                    a.click();
                """
            )
        )
    )

# Function to run the web server
def run_web_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """
    Run the SimNexus web server.
    
    Args:
        host: The host to bind to
        port: The port to bind to
        reload: Whether to enable auto-reload for development
    """
    serve(app=app, host=host, port=port, reload=reload)

# Main entry point for running the app directly (for development)
if __name__ == "__main__":
    run_web_server(reload=True)