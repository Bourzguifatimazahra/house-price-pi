.PHONY: help install setup clean data features train predict test lint format dashboard docker-build docker-run ci-cd

# Colors for help
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM=20

## Show help
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

## Install dependencies
install:
	pip install --upgrade pip
	pip install -e .[dev]
	pre-commit install

## Setup project structure
setup:
	mkdir -p data/raw data/processed data/external
	mkdir -p artifacts/models artifacts/metrics artifacts/predictions artifacts/logs artifacts/exports
	mkdir -p artifacts/exports/par_ville
	mkdir -p logs
	touch data/raw/.gitkeep data/processed/.gitkeep data/external/.gitkeep
	@echo "✅ Project structure created"

## Clean generated files
clean:
	rm -rf artifacts/*
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "✅ Clean completed"

## Generate sample data
data:
	python scripts/generate_data.py --samples 1000 --output data/raw/washington_real_estate.csv

## Run feature engineering
features:
	python -c "from src.features.build_features import main; main('data/raw/washington_real_estate.csv', 'data/processed/processed_data.csv')"

## Train models (all)
train:
	python -c "from src.models.train import main; main()"

## Train quantile models
train-quantile:
	python -c "from src.models.quantile_trainer import main; main()"

## Generate predictions
predict:
	python -c "from src.models.predict import main; main()"

## Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

## Run quick tests (smoke)
test-smoke:
	pytest tests/ -m smoke -v

## Run linting
lint:
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy src/ --ignore-missing-imports

## Format code
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

## Launch dashboard
dashboard:
	streamlit run dashboard/app.py

## Build Docker image
docker-build:
	docker build -t house-price-pi:latest .

## Run Docker container
docker-run:
	docker run -p 8501:8501 -v $(PWD)/artifacts:/app/artifacts house-price-pi:latest

## Run CI/CD pipeline locally
ci-cd:
	@echo "${YELLOW}Running CI pipeline...${RESET}"
	make lint
	make test
	@echo "${YELLOW}Running CD pipeline...${RESET}"
	make data
	make features
	make train
	make predict
	@echo "${GREEN}✅ CI/CD completed successfully${RESET}"

## Export predictions by city
export:
	python scripts/export_predictions.py

## Generate evaluation report
report:
	python -c "from src.evaluation.visualizer import generate_report; generate_report()"

## Run complete pipeline
pipeline:
	make clean
	make data
	make features
	make train-quantile
	make predict
	make export
	make report
	@echo "${GREEN}✅ Complete pipeline executed${RESET}"