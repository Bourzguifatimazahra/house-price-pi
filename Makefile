.PHONY: help install setup train test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  setup       Setup project structure"
	@echo "  train       Train models"
	@echo "  predict     Make predictions"
	@echo "  test        Run tests"
	@echo "  lint        Run linters"
	@echo "  format      Format code"
	@echo "  clean       Clean artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

install:
	pip install -e .[dev]

setup:
	mkdir -p data/raw data/processed data/external
	mkdir -p artifacts/models artifacts/metrics artifacts/predictions
	mkdir -p notebooks artifacts logs
	python scripts/setup.py

train:
	python src/models/train.py

predict:
	python src/models/predict.py

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf artifacts/
	rm -rf data/processed/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

docker-build:
	docker build -t house-price-pi .

docker-run:
	docker run -p 8501:8501 house-price-pi