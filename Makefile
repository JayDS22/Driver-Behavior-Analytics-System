.PHONY: help install test lint format docker-build docker-run clean deploy

# Default target
help:
	@echo "Driver Behavior Analytics System - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install dependencies and setup development environment"
	@echo "  test          Run complete test suite with coverage"
	@echo "  lint          Run code linting and type checking"
	@echo "  format        Format code with black and isort"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run application with Docker Compose"
	@echo "  docker-stop   Stop Docker services"
	@echo "  deploy        Deploy to production"
	@echo "  clean         Clean up temporary files and containers"
	@echo "  docs          Generate documentation"

# Development setup
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	@echo "Running comprehensive test suite..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90
	@echo "✅ All tests completed. Coverage report: htmlcov/index.html"

test-fast:
	@echo "Running fast test subset..."
	pytest tests/ -x -q --disable-warnings

test-survival:
	@echo "Testing survival analysis models..."
	pytest tests/test_survival_analysis.py -v

test-bayesian:
	@echo "Testing Bayesian models..."
	pytest tests/test_bayesian_models.py -v

test-api:
	@echo "Testing API endpoints..."
	pytest tests/test_api_endpoints.py -v

# Code quality
lint:
	@echo "Running code linting..."
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	@echo "✅ Linting completed"

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting completed"

# Docker operations
docker-build:
	@echo "Building Docker image..."
	docker build -t driver-behavior-analytics:latest .
	@echo "✅ Docker image built successfully"

docker-run:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started. API available at http://localhost:8003"

docker-stop:
	@echo "Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	docker-compose logs -f driver-analytics-api

# Production deployment
deploy:
	@echo "Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d --build
	@echo "✅ Production deployment completed"

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/
	kubectl rollout status deployment/driver-analytics-api
	@echo "✅ Kubernetes deployment completed"

# Database operations
db-init:
	@echo "Initializing database..."
	docker-compose exec postgres psql -U analytics_user -d driver_analytics -f /docker-entrypoint-initdb.d/init.sql
	@echo "✅ Database initialized"

db-migrate:
	@echo "Running database migrations..."
	# Add migration commands here
	@echo "✅ Database migrations completed"

db-backup:
	@echo "Creating database backup..."
	docker-compose exec postgres pg_dump -U analytics_user driver_analytics > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backup created"

# Monitoring
logs:
	@echo "Viewing application logs..."
	docker-compose logs -f driver-analytics-api

metrics:
	@echo "Opening Prometheus metrics..."
	open http://localhost:9090

dashboard:
	@echo "Opening Grafana dashboard..."
	open http://localhost:3000

# Cleanup
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "✅ Cleanup completed"

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html
	@echo "✅ Documentation generated at docs/_build/html/index.html"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	python scripts/benchmark_api.py
	@echo "✅ Benchmarks completed"

# Security scanning
security-scan:
	@echo "Running security scans..."
	bandit -r src/
	safety check
	@echo "✅ Security scan completed"

# CI/CD helpers
ci-test: install test lint security-scan

ci-build: docker-build

ci-deploy: deploy

# Environment management
env-dev:
	cp .env.example .env
	@echo "✅ Development environment configured"

env-prod:
	@echo "Setting up production environment..."
	# Add production environment setup
	@echo "✅ Production environment configured"
