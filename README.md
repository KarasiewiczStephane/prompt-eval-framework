# Prompt Engineering Evaluation Framework

A testing suite for LLM prompts with metrics tracking, A/B testing, cost optimization, and reporting.

## Features

- **Multi-Model Evaluation** &mdash; Test prompts across GPT-4, GPT-3.5, Claude Sonnet, Claude Haiku
- **Comprehensive Metrics** &mdash; Accuracy, latency (p50/p95/p99), token usage, cost
- **A/B Testing** &mdash; Statistical comparison with McNemar's test and bootstrap CI
- **Cost Optimization** &mdash; Estimation, budget limits, cheaper model recommendations
- **HTML Reports** &mdash; Interactive charts with Plotly
- **Version Control** &mdash; Auto-versioning for prompt iterations
- **DuckDB Storage** &mdash; Full traceability of all runs

## Quick Start

```bash
# Clone and install
git clone https://github.com/KarasiewiczStephane/prompt-eval-framework.git
cd prompt-eval-framework
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run your first evaluation
python -m src.cli run --suite example_suites/greeting_suite.yaml --model gpt-4

# Generate HTML report
python -m src.cli report --run-id 1 --output report.html
```

## Usage

### Running Evaluations

```bash
# Single model
python -m src.cli run --suite my_suite.yaml --model gpt-4

# Multiple models
python -m src.cli run --suite my_suite.yaml --model gpt-4 --model claude-sonnet

# Filter by tags
python -m src.cli run --suite my_suite.yaml --model gpt-4 --tags edge-case

# With budget limit
python -m src.cli run --suite my_suite.yaml --model gpt-4 --budget 1.00
```

### A/B Testing

```bash
python -m src.cli compare --a prompts/v1.yaml --b prompts/v2.yaml \
  --suite tests.yaml --model gpt-4
```

### Cost Estimation

```bash
python -m src.cli estimate --suite my_suite.yaml --model gpt-4 --model gpt-3.5
```

### Version History

```bash
python -m src.cli history --prompt greeting
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

## Docker

```bash
# Build
docker build -t prompteval .

# Run
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  prompteval run --suite /app/suites/test.yaml --model gpt-4

# Development
docker-compose run --rm dev
```

## Project Structure

```
prompt-eval-framework/
├── src/
│   ├── cli.py                  # Click CLI
│   ├── prompts/                # Template management & versioning
│   ├── testing/                # Assertions & test runner
│   ├── evaluation/             # Model runners, metrics, A/B testing
│   ├── reporting/              # HTML report generator
│   └── utils/                  # Config, DB, logging, history
├── example_prompts/            # Sample prompt templates
├── example_suites/             # Sample test suites
├── tests/                      # Unit tests (235+)
├── configs/config.yaml         # Default configuration
├── .github/workflows/ci.yml   # CI pipeline
├── Dockerfile                  # Production container
├── docker-compose.yml          # Development setup
└── Makefile                    # Common commands
```

## Development

```bash
# Install dependencies
make install

# Run tests
make test

# Lint
make lint

# Clean
make clean
```

## License

MIT
