pip-sync:
	uv sync --default-index https://pypi.org/simple

format:
	tox -e lint
	tox -e format

format-nb:
	tox -e lint-nb
	tox -e format-nb

check:
	tox -e check-lint-types 
	tox -e check-formatting	

test:
	tox -e test

tox:
	tox


bumpver-patch:
	bumpver update --patch
