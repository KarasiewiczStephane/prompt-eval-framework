.PHONY: install test lint clean run docker-build docker-run docker-dev

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

run:
	python -m src.cli $(ARGS)

docker-build:
	docker build -t prompteval .

docker-run:
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/example_prompts:/app/prompts \
		-v $(PWD)/example_suites:/app/suites \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		prompteval $(ARGS)

docker-dev:
	docker-compose run --rm dev
