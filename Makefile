pip-sync:
	uv sync

format:
	tox -e lint
	tox -e format

format-nb:
	ruff check --extend-select I --fix notebooks
	ruff format notebooks

check:
	tox -e check-lint-types 
	tox -e check-formatting	

test:
	tox -e test

tox:
	tox


bumpver-patch:
	bumpver update --patch
