.PHONY: train api dashboard test docker down clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

train: ## Run training pipeline locally
	python train.py

api: ## Start API server locally
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard locally
	streamlit run dashboard/app.py

test: ## Run all tests
	python -m pytest tests/ -v

test-unit: ## Run unit tests only
	python -m pytest tests/test_unit.py -v

test-integration: ## Run integration tests (requires API running)
	python -m pytest tests/test_integration.py -v

docker: ## Start all services with Docker Compose
	docker compose up --build -d

down: ## Stop all Docker services
	docker compose down

logs: ## Tail Docker logs
	docker compose logs -f

clean: ## Remove model artifacts and processed data
	rm -rf models/*.pkl models/*.pt models/*.json
	rm -rf data/processed/*
	rm -rf mlruns/ logs/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
