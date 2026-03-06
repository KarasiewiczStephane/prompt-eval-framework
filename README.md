# Prompt Engineering Evaluation Framework

A testing suite for LLM prompts with metrics tracking, A/B testing, cost optimization, and reporting.

## Features

- **Multi-Model Evaluation** -- Test prompts across GPT-4, GPT-3.5, Claude Sonnet, Claude Haiku
- **Comprehensive Metrics** -- Accuracy, latency (p50/p95/p99), token usage, cost
- **A/B Testing** -- Statistical comparison with McNemar's test and bootstrap CI
- **Cost Optimization** -- Estimation, budget limits, cheaper model recommendations
- **HTML Reports** -- Interactive charts with Plotly
- **Version Control** -- Auto-versioning for prompt iterations
- **DuckDB Storage** -- Full traceability of all runs
- **Streamlit Dashboard** -- Interactive visualization of evaluation results, model comparisons, and cost analysis

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/KarasiewiczStephane/prompt-eval-framework.git
cd prompt-eval-framework
make install

# 2. Configure API keys
cp .env.example .env
# Edit .env with your actual keys:
#   OPENAI_API_KEY="sk-..."
#   ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run an evaluation
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4

# 4. Launch the dashboard
make dashboard
# Opens at http://localhost:8501
```

## CLI Usage

The CLI is built with Click. All commands are invoked via `python -m src.cli`.

### Running Evaluations

```bash
# Single model
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4

# Multiple models
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4 --model claude-sonnet

# Filter by tags
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4 --tags basic

# With budget limit
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4 --budget 1.00

# Export results to JSON
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4 --output results.json
```

You can also use the Makefile shorthand:

```bash
make run ARGS="run --suite example_suites/greeting_suite.yaml --model gpt-4"
```

### A/B Testing

```bash
python -m src.cli compare \
  --a example_prompts/greeting.yaml \
  --b example_prompts/summarization.yaml \
  --suite example_suites/greeting_suite.yaml \
  --model gpt-4
```

### Cost Estimation

```bash
python -m src.cli estimate --suite example_suites/greeting_suite.yaml --model gpt-4 --model gpt-3.5-turbo
```

### Version History

```bash
python -m src.cli history --prompt greeting
```

### Generate Report

```bash
python -m src.cli report --run-id 1 --output report.html
```

## Dashboard

The Streamlit dashboard provides interactive visualization of evaluation metrics using synthetic demo data:

- Summary metrics (total tests, best accuracy, models tested, total cost)
- Model accuracy comparison bar chart
- Latency analysis (avg and p95) per model
- Cost breakdown pie chart
- Accuracy by test category
- A/B test results table

Launch it with:

```bash
make dashboard
# or directly:
streamlit run src/dashboard/app.py
```

## Prompt Template Format

```yaml
# example_prompts/greeting.yaml
name: greeting
category: customer-service
system_prompt: You are a helpful customer service agent.
user_prompt: |
  Greet the customer named {{ customer_name }} who is inquiring about {{ topic }}.
variables:
  - customer_name
  - topic
model_config:
  temperature: 0.7
  max_tokens: 256
few_shot_examples:
  - user: "Greet John asking about returns"
    assistant: "Hello John! Thank you for reaching out..."
```

## Test Suite Format

```yaml
# example_suites/greeting_suite.yaml
name: greeting_test_suite
prompt: greeting
test_cases:
  - id: tc_basic
    name: Basic greeting
    input:
      customer_name: John
      topic: returns
    expected: "Hello John"
    assertion: contains
    tags: [basic]
```

Four example suites are included: `greeting_suite.yaml`, `classification_suite.yaml`, `extraction_suite.yaml`, and `summarization_suite.yaml`.

## Configuration

Default configuration lives in `configs/config.yaml`. Key settings:

- **database.path** -- DuckDB storage location (`data/prompteval.duckdb`)
- **defaults** -- Default model parameters (temperature, max_tokens, timeout)
- **paths** -- Directories for prompts, suites, and reports
- **pricing** -- Per-model token pricing for cost estimation

Environment variables (via `.env`) override config values. See `.env.example` for required keys.

## Docker

```bash
# Build and run
docker build -t prompteval .
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  prompteval run --suite /app/suites/greeting_suite.yaml --model gpt-4

# Development shell
docker-compose run --rm dev
```

## Project Structure

```
prompt-eval-framework/
├── src/
│   ├── cli.py                  # Click CLI (run, compare, report, estimate, history)
│   ├── prompts/                # Template management, versioning, variable resolution
│   ├── testing/                # Assertions & test runner
│   ├── evaluation/             # Model runners, metrics, A/B testing, cost optimizer
│   ├── reporting/              # HTML report generator
│   ├── dashboard/              # Streamlit visualization app
│   └── utils/                  # Config, DuckDB, logging, history
├── example_prompts/            # Sample prompt templates (5 prompts)
├── example_suites/             # Sample test suites (4 suites)
├── tests/                      # Unit tests (280+)
├── configs/config.yaml         # Default configuration
├── .github/workflows/ci.yml   # CI pipeline (lint, test, docker)
├── Dockerfile                  # Production container
├── docker-compose.yml          # Development setup
└── Makefile                    # Common commands
```

## Development

```bash
make install      # Install dependencies
make test         # Run tests with coverage
make lint         # Ruff check and format
make clean        # Remove cache files
make run ARGS="--help"   # Run CLI with arguments
make dashboard    # Launch Streamlit dashboard
```

## License

MIT
