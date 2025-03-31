# Development Guide

This document explains how to set up your development environment and workflow for contributing to SimNexus.

## Development Environment

### Prerequisites

- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) for dependency management
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/teaching-repositories/simnexus.git
   cd simnexus
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .[dev]
   ```

   This will install the package in development mode, making the package editable while working on it.

## Development Workflow

### Code Style and Quality

SimNexus uses:
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- [MyPy](https://mypy.readthedocs.io/) for type checking

Run linting:
```bash
ruff check .
```

Run formatting:
```bash
ruff format .
```

Run type checking:
```bash
mypy simnexus
```

### Testing

SimNexus uses [pytest](https://docs.pytest.org/) for testing.

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=simnexus
```

### Documentation

SimNexus uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

Build documentation:
```bash
mkdocs build
```

Serve documentation locally:
```bash
mkdocs serve
```

## Project Structure

```
simnexus/
├── src/                   # Source directory
│   ├── simnexus/          # Package source code
│   │   ├── core/          # Core simulation logic
│   │   ├── cli/           # Command line interface
│   │   ├── tui/           # Terminal user interface
│   │   ├── web/           # Web interface
│   │   ├── viz/           # Visualization utilities
│   │   ├── utils/         # Helper utilities
│   │   └── config/        # Configuration management
│
├── tests/                 # Test suite
├── docs/                  # Documentation
├── examples/              # Example scripts
├── scripts/               # Build and utility scripts
├── .github/               # GitHub workflows and templates
├── pyproject.toml         # Project configuration
└── mkdocs.yml             # Documentation configuration
```

## Releasing

1. Update version in `pyproject.toml`
2. Update the changelog
3. Run the release script:
   ```bash
   # Install required dependencies if needed
   pip install build twine
   
   # Test the release (uploads to TestPyPI)
   python scripts/release.py --test
   
   # Make the actual release
   python scripts/release.py
   ```

Alternatively, you can still create and push tags manually:
```bash
git tag -a v0.x.x -m "Release v0.x.x"
git push origin v0.x.x
```

## Continuous Integration

GitHub Actions workflows automatically run:
- Tests on multiple Python versions
- Linting and type checking
- Documentation building and deployment

See `.github/workflows/` for details.