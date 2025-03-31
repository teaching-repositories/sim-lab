"""Web interface for SimNexus using FastHTML."""

from fasthtml import FastHTML, Route, get, post
from fasthtml.responses import HTMLResponse
from fasthtml.templates import Jinja2Templates
from pathlib import Path
import os
import numpy as np
import json
from typing import List, Optional

from simnexus import (
    StockMarketSimulation,
    ResourceFluctuationsSimulation,
    ProductPopularitySimulation
)

# Set up templates directory
templates_dir = Path(__file__).parent / "templates"
if not templates_dir.exists():
    templates_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create base template file if it doesn't exist
    base_template = templates_dir / "base.html"
    if not base_template.exists():
        with open(base_template, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SimNexus{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/htmx.org@1.9.2"></script>
    <style>
        body { padding-top: 2rem; padding-bottom: 2rem; }
        .navbar { margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        <header class="d-flex flex-wrap justify-content-center py-3 mb-4 border-bottom">
            <a href="/" class="d-flex align-items-center mb-3 mb-md-0 me-md-auto text-dark text-decoration-none">
                <span class="fs-4">SimNexus</span>
            </a>
            <ul class="nav nav-pills">
                <li class="nav-item"><a href="/" class="nav-link active" aria-current="page">Home</a></li>
                <li class="nav-item"><a href="/stock" class="nav-link">Stock Market</a></li>
                <li class="nav-item"><a href="/resource" class="nav-link">Resource Fluctuations</a></li>
                <li class="nav-item"><a href="/product" class="nav-link">Product Popularity</a></li>
            </ul>
        </header>
        <main>
            {% block content %}{% endblock %}
        </main>
        <footer class="pt-5 my-5 text-muted border-top">
            SimNexus &middot; &copy; 2025
        </footer>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

templates = Jinja2Templates(directory=str(templates_dir))

# Create FastHTML app
app = FastHTML()

@app.get("/", response_class=HTMLResponse)
async def index(request):
    """Render the home page."""
    return templates.TemplateResponse(
        "base.html",
        {"request": request, "block_content": "<h1>Welcome to SimNexus</h1><p>Select a simulation from the navigation menu above.</p>"}
    )

def create_app():
    """Create and configure the FastHTML application."""
    return app