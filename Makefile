pip-sync:
	uv sync

format:
	uv run ruff check --extend-select I --fix
	uv run ruff format src tests

test-coverage:
	uv run pytest --cov-report=html --cov=src tests
