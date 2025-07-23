.PHONY: help build up down restart logs test lint format clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker containers
	docker-compose build

up: ## Start development environment
	docker-compose up -d

down: ## Stop development environment
	docker-compose down

restart: ## Restart development environment
	docker-compose restart

logs: ## Show container logs
	docker-compose logs -f

test: ## Run tests
	docker-compose exec ml-dev pytest tests/ -v

test-data: ## Test data processing functions specifically
	docker-compose exec ml-dev pytest tests/test_data.py -v

test-coverage: ## Run tests with coverage report
	docker-compose exec ml-dev pytest tests/ --cov=src --cov-report=html --cov-report=term
	
lint: ## Run code linting
	docker-compose exec ml-dev flake8 src/ tests/

format: ## Format code
	docker-compose exec ml-dev black src/ tests/

clean: ## Clean up containers and images
	docker-compose down --rmi all --volumes --remove-orphans

shell: ## Open shell in container
	docker-compose exec ml-dev bash

notebook: ## Start Jupyter notebook
	@echo "Jupyter notebook available at: http://localhost:8888"
	@echo "MLflow UI available at: http://localhost:5000"

setup-data: ## Download and setup initial data
	docker-compose exec ml-dev python src/data/load_data.py