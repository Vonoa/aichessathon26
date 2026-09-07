SHELL := /bin/bash

.PHONY: setup play arena zip gate test bench

setup:
	uv sync

test:
	uv run pytest -q

bench:
	uv run python -m tools.bench

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run pytest -q
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
