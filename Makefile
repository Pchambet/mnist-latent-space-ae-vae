.PHONY: setup data run report test lint all clean

setup:  ## install the locked environment
	uv sync --locked

data:  ## download MNIST into data/raw (cached, ~65 MB)
	uv run mnist-latent data

run:  ## train every model (checkpoints cached), evaluate, draw figures
	uv run mnist-latent run

report:  ## build site/index.html and refresh the README result tables
	uv run mnist-latent report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: setup data run report

clean:  ## remove cached checkpoints and arrays (forces retraining)
	rm -rf data/interim
