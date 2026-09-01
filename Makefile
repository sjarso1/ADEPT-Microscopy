.PHONY: install test lint format smoke build clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check . --fix

smoke:
	bash scripts/smoke_e2e.sh

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache results
