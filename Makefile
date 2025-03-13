pip-sync:
	uv sync

format:
	uv run ruff check --extend-select I --fix src tests
	uv run ruff format src tests

format-nb:
	uv run ruff check --extend-select I --fix notebooks
	uv run ruff format notebooks


check: format
	tox -e lint -e lint-types	

test-coverage:
	uv run pytest --cov-report=html --cov=src tests
