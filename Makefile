SHELL := /bin/bash

.PHONY: setup play arena zip gate test

setup:
	uv sync

test:
	uv run pytest -q

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run pytest -q
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
