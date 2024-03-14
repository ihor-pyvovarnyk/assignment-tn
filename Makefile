activate_venv:
	poetry shell

install:
	poetry install

lint:
	flake8 assignment/
	isort --check-only --diff --stdout .
	black --diff .

format:
	isort .
	black .
