# Contributing to edRFT

Thank you for your interest in contributing to edRFT.

## How to contribute

You can help by:

- reporting bugs,
- improving documentation,
- adding examples,
- improving tests,
- suggesting API improvements, or
- contributing model and forecasting utilities.

## Development setup

```bash
git clone https://github.com/statsdl/edRFT.git
cd edRFT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Pull request checklist

Before opening a pull request, please check:

```bash
pytest
python -m build
python -m twine check dist/*
```

## Style

Please keep examples simple, documented, and reproducible. New public functions should include clear docstrings and tests where possible.

## Issues

For bugs, include:

- Python version,
- operating system,
- edRFT version,
- minimal code to reproduce the issue,
- full error message or traceback.

## License

By contributing, you agree that your contribution will be distributed under the MIT License.
