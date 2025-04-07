"""Web interface for SimLab using FastHTML."""

from fasthtml import FastHTML, Route, get, post
from fasthtml.responses import HTMLResponse, JSONResponse
from fasthtml.templates import Jinja2Templates
from pathlib import Path
import os
import numpy as np
import json
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import io
import base64

from sim_lab import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)

# Set up templates directory
templates_dir = Path(__file__).parent / "templates"
if not templates_dir.exists():
    os.makedirs(templates_dir, exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Create FastHTML app
app = FastHTML()

@app.get("/", response_class=HTMLResponse)
async def index(request):
    """Render the home page."""
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "active_page": "home"}
    )

@app.get("/stock", response_class=HTMLResponse)
async def stock_market(request):
    """Render the stock market simulation page."""
    return templates.TemplateResponse(
        "stock.html",
        {"request": request, "active_page": "stock"}
    )

@app.post("/api/stock/run", response_class=HTMLResponse)
async def run_stock_simulation(
    request,
    days: int = 365,
    start_price: float = 100.0,
    volatility: float = 2.0,
    drift: float = 0.1,
    enable_event: bool = False,
    event_day: Optional[int] = None,
    event_impact: Optional[float] = None,
    random_seed: Optional[int] = None,
):
    """Run a stock market simulation with the provided parameters."""
    # Convert parameters
    volatility = float(volatility) / 100  # Convert from percentage to decimal
    drift = float(drift) / 100  # Convert from percentage to decimal
    
    # Handle event parameters
    if enable_event and event_day is not None and event_impact is not None:
        event_day = int(event_day)
        event_impact = float(event_impact) / 100  # Convert from percentage to decimal
    else:
        event_day = None
        event_impact = None
    
    # Create and run simulation
    sim = StockMarketSimulation(
        start_price=start_price,
        days=days,
        volatility=volatility,
        drift=drift,
        event_day=event_day,
        event_impact=event_impact,
        random_seed=random_seed,
    )
    
    prices = sim.run_simulation()
    
    # Calculate key statistics
    final_price = prices[-1]
    change_pct = ((final_price / start_price) - 1) * 100
    max_price = max(prices)
    min_price = min(prices)
    max_day = prices.index(max_price)
    min_day = prices.index(min_price)
    
    # Generate chart
    plt.figure(figsize=(10, 6))
    plt.plot(prices)
    plt.title("Stock Market Simulation")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha=0.3)
    
    if event_day is not None:
        plt.axvline(x=event_day, color='red', linestyle='--', 
                   label=f'Market Event (Impact: {event_impact*100:.1f}%)')
        plt.legend()
    
    # Save chart to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return results as HTML
    html_content = f"""
    <div class="mb-4">
        <img src="data:image/png;base64,{chart_data}" class="img-fluid" alt="Stock Price Chart">
    </div>
    
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card bg-light">
                <div class="card-body">
                    <h5 class="card-title">Key Statistics</h5>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Starting Price:</span>
                            <span>${start_price:.2f}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Final Price:</span>
                            <span>${final_price:.2f}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Change:</span>
                            <span class="{('text-success' if change_pct >= 0 else 'text-danger')}">{change_pct:.2f}%</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card bg-light">
                <div class="card-body">
                    <h5 class="card-title">Extremes</h5>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Maximum Price:</span>
                            <span>${max_price:.2f} (Day {max_day})</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Minimum Price:</span>
                            <span>${min_price:.2f} (Day {min_day})</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Volatility:</span>
                            <span>{volatility*100:.2f}%</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <div class="d-flex justify-content-between">
        <button class="btn btn-primary" onclick="document.getElementById('simulation-form').dispatchEvent(new Event('submit'))">Run Again</button>
        <button class="btn btn-outline-secondary" onclick="window.open('/api/stock/download?data=' + encodeURIComponent('{','.join([str(p) for p in prices])}'))">Download CSV</button>
    </div>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/stock/download")
async def download_stock_data(request, data: str):
    """Generate a CSV file of simulation data."""
    prices = [float(p) for p in data.split(',')]
    csv_content = "day,price\n" + "\n".join([f"{i},{price}" for i, price in enumerate(prices)])
    
    return HTMLResponse(
        content=csv_content,
        headers={
            "Content-Disposition": "attachment; filename=stock_simulation.csv",
            "Content-Type": "text/csv",
        }
    )

@app.get("/resource", response_class=HTMLResponse)
async def resource_fluctuations(request):
    """Placeholder for resource fluctuations page."""
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request, 
            "active_page": "resource",
            "block_content": "<h1>Resource Fluctuations Simulation</h1><p>Coming soon! This page is under development.</p>"
        }
    )

@app.get("/product", response_class=HTMLResponse)
async def product_popularity(request):
    """Placeholder for product popularity page."""
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request, 
            "active_page": "product",
            "block_content": "<h1>Product Popularity Simulation</h1><p>Coming soon! This page is under development.</p>"
        }
    )

@app.get("/docs", response_class=HTMLResponse)
async def documentation(request):
    """Placeholder for documentation page."""
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request, 
            "active_page": "docs",
            "block_content": "<h1>SimLab Documentation</h1><p>Coming soon! Comprehensive documentation is under development.</p>"
        }
    )

def create_app():
    """Create and configure the FastHTML application."""
    return app