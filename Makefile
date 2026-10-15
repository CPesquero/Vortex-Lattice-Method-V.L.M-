.PHONY: install dev test lint format figures reproduce notebook clean

install:            ## install the package
	pip install .

dev:                ## editable install with test and lint tools
	pip install -e ".[dev]"

test:               ## run the test suite
	pytest

lint:               ## static checks
	ruff check .
	ruff format --check .

format:             ## auto-format
	ruff format .
	ruff check --fix .

figures:            ## regenerate every figure in figures/ and results/results.json
	vlmwing --figures-dir figures --json results/results.json

reproduce: figures  ## alias

notebook:           ## open the walkthrough notebook
	jupyter notebook notebooks/vlm_analysis.ipynb

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
