pip-sync:
	uv sync

format:
	tox -e lint
	tox -e format

format-nb:
	uv run ruff check --extend-select I --fix notebooks
	uv run ruff format notebooks

check: format
	tox -e lint -e lint-types -e check-formatting	

test:
	tox -e test

tox:
	tox


bumpver-patch:
	uv run bumpver update --patch
