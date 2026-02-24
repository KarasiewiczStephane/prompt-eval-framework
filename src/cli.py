"""Click-based CLI for the prompt evaluation framework.

Provides commands: run, compare, report, estimate, and history.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.evaluation.ab_tester import ABTester
from src.evaluation.cost_optimizer import CostOptimizer
from src.evaluation.metrics import MetricsCalculator, TestResult, compare_models
from src.evaluation.model_runner import MultiModelRunner
from src.prompts.template_manager import TemplateManager
from src.prompts.variable_resolver import VariableResolver, build_messages
from src.prompts.version_control import VersionControl
from src.testing.assertions import assertion_factory
from src.testing.test_runner import TestSuite
from src.utils.config import Config
from src.utils.database import Database
from src.utils.logger import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Prompt Engineering Evaluation Framework."""
    ctx.ensure_object(dict)
    cfg = Config.from_yaml(Path(config)) if config else Config.from_env()
    setup_logging(cfg.log_level)
    ctx.obj["config"] = cfg


@cli.command()
@click.option(
    "--suite",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Test suite YAML file",
)
@click.option(
    "--model", "-m", multiple=True, default=["gpt-4"], help="Model(s) to evaluate"
)
@click.option("--tags", "-t", multiple=True, help="Filter test cases by tags")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--budget", "-b", type=float, help="Max budget in USD")
@click.pass_context
def run(
    ctx: click.Context,
    suite: str,
    model: tuple[str, ...],
    tags: tuple[str, ...],
    output: str | None,
    budget: float | None,
) -> None:
    """Run evaluation on a test suite."""
    cfg: Config = ctx.obj["config"]

    try:
        test_suite = TestSuite.from_yaml(Path(suite))

        if tags:
            test_suite.test_cases = test_suite.filter_by_tags(include=list(tags))

        console.print(
            f"[bold]Running {len(test_suite.test_cases)} tests "
            f"on {len(model)} model(s)[/bold]"
        )

        if budget:
            optimizer = CostOptimizer(cfg.pricing or None)
            prompts = [
                _render_prompt(cfg, test_suite.prompt_name, tc.input_variables)
                for tc in test_suite.test_cases
            ]
            estimate = optimizer.estimate_cost(prompts, list(model), budget=budget)
            if not estimate.within_budget:
                console.print(
                    f"[red]Estimated cost ${estimate.estimated_cost_usd:.2f} "
                    f"exceeds budget ${budget:.2f}[/red]"
                )
                if not click.confirm("Continue anyway?"):
                    sys.exit(1)

        results = asyncio.run(_run_evaluation(cfg, test_suite, list(model)))

        _display_results(results)

        if output:
            _export_results(results, Path(output))
            console.print(f"Results exported to {output}")

    except FileNotFoundError as exc:
        console.print(f"[red]File not found: {exc}[/red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        logger.exception("Run failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--a",
    "variant_a",
    required=True,
    type=click.Path(exists=True),
    help="Variant A prompt file",
)
@click.option(
    "--b",
    "variant_b",
    required=True,
    type=click.Path(exists=True),
    help="Variant B prompt file",
)
@click.option(
    "--suite",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Test suite to run",
)
@click.option("--model", "-m", default="gpt-4", help="Model to use")
@click.pass_context
def compare(
    ctx: click.Context,
    variant_a: str,
    variant_b: str,
    suite: str,
    model: str,
) -> None:
    """A/B test two prompt variants."""
    cfg: Config = ctx.obj["config"]

    try:
        db = Database(cfg.db_path)
        mgr_a = TemplateManager(db, Path(variant_a).parent)
        mgr_b = TemplateManager(db, Path(variant_b).parent)
        prompt_a = mgr_a.load(Path(variant_a).stem)
        prompt_b = mgr_b.load(Path(variant_b).stem)
        test_suite = TestSuite.from_yaml(Path(suite))

        console.print(f"[bold]A/B Test: {variant_a} vs {variant_b}[/bold]")

        results_a, results_b = asyncio.run(
            _run_ab_variants(cfg, prompt_a, prompt_b, test_suite, model)
        )

        tester = ABTester()
        result = tester.compare(
            [r.passed for r in results_a],
            [r.passed for r in results_b],
        )

        table = Table(title="A/B Test Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Variant A Accuracy", f"{result.variant_a_accuracy:.1%}")
        table.add_row("Variant B Accuracy", f"{result.variant_b_accuracy:.1%}")
        table.add_row("Winner", result.winner)
        table.add_row("P-Value", f"{result.p_value:.4f}")
        table.add_row("Effect Size", f"{result.effect_size:+.3f}")
        table.add_row("95% CI", f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
        table.add_row("Recommendation", result.recommendation)
        console.print(table)

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        logger.exception("Compare failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--run-id",
    "-r",
    required=True,
    type=int,
    help="Run ID to generate report for",
)
@click.option("--output", "-o", default="report.html", help="Output HTML file")
@click.pass_context
def report(ctx: click.Context, run_id: int, output: str) -> None:
    """Generate HTML report for a run."""
    cfg: Config = ctx.obj["config"]

    try:
        db = Database(cfg.db_path)
        with db.connection() as conn:
            run_data = conn.execute(
                "SELECT * FROM runs WHERE id = ?", [run_id]
            ).fetchone()

        if not run_data:
            console.print(f"[red]Run {run_id} not found[/red]")
            sys.exit(1)

        console.print(
            f"[green]Report generation for run {run_id} — "
            f"use task-10 for full HTML reports[/green]"
        )

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--suite",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Test suite file",
)
@click.option(
    "--model", "-m", multiple=True, default=["gpt-4"], help="Model(s) to estimate"
)
@click.pass_context
def estimate(ctx: click.Context, suite: str, model: tuple[str, ...]) -> None:
    """Estimate cost for running a test suite."""
    cfg: Config = ctx.obj["config"]
    test_suite = TestSuite.from_yaml(Path(suite))
    optimizer = CostOptimizer(cfg.pricing or None)

    prompts = ["Sample prompt text"] * len(test_suite.test_cases)
    est = optimizer.estimate_cost(prompts, list(model))

    table = Table(title="Cost Estimate")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Test Cases", str(len(test_suite.test_cases)))
    table.add_row("Est. Input Tokens", f"{est.estimated_input_tokens:,}")
    table.add_row("Est. Output Tokens", f"{est.estimated_output_tokens:,}")
    table.add_row("Est. Total Cost", f"${est.estimated_cost_usd:.4f}")
    for m, cost in est.breakdown_by_model.items():
        table.add_row(f"  {m}", f"${cost:.4f}")
    console.print(table)


@cli.command()
@click.option("--prompt", "-p", required=True, help="Prompt name")
@click.option("--limit", "-l", default=10, help="Number of versions to show")
@click.pass_context
def history(ctx: click.Context, prompt: str, limit: int) -> None:
    """Show version history for a prompt."""
    cfg: Config = ctx.obj["config"]
    db = Database(cfg.db_path)
    vc = VersionControl(db)

    versions = vc.get_history(prompt)[:limit]

    if not versions:
        console.print(f"[yellow]No history found for prompt '{prompt}'[/yellow]")
        return

    table = Table(title=f"Version History: {prompt}")
    table.add_column("Version", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Hash", style="dim")

    for v in versions:
        table.add_row(str(v["version"]), str(v["created_at"]), str(v["hash"])[:8])

    console.print(table)


# ── Internal helpers ────────────────────────────────────────────────


def _render_prompt(cfg: Config, prompt_name: str, variables: dict) -> str:
    """Render a prompt and return the user-message text."""
    db = Database(cfg.db_path)
    mgr = TemplateManager(db, cfg.prompts_dir)
    resolver = VariableResolver()
    template = mgr.load(prompt_name)
    result = resolver.render(template, variables)
    return result.user_prompt


async def _run_evaluation(
    cfg: Config,
    test_suite: TestSuite,
    models: list[str],
) -> dict[str, list[TestResult]]:
    """Run all test cases across all models and return results."""
    runner = MultiModelRunner(cfg)
    db = Database(cfg.db_path)
    mgr = TemplateManager(db, cfg.prompts_dir)
    resolver = VariableResolver()
    template = mgr.load(test_suite.prompt_name)
    all_results: dict[str, list[TestResult]] = {}

    for model in models:
        model_results: list[TestResult] = []
        for tc in test_suite.test_cases:
            try:
                rendered = resolver.render(template, tc.input_variables)
                messages = build_messages(rendered)
                response = await runner.run_single(
                    messages, template.model_config, model
                )
                assertion = assertion_factory(
                    {"type": tc.assertion_type, "params": tc.assertion_params}
                )
                check = assertion.evaluate(response.content, tc.expected_output)
                model_results.append(
                    TestResult(
                        passed=check.passed,
                        response=response,
                        latency_ms=response.latency_ms,
                        test_case=tc,
                    )
                )
            except Exception as exc:
                logger.error("Test %s failed: %s", tc.id, exc)
                model_results.append(TestResult(passed=False, test_case=tc))
        all_results[model] = model_results

    return all_results


async def _run_ab_variants(cfg, prompt_a, prompt_b, test_suite, model):
    """Run both A/B variants and return paired results."""
    runner = MultiModelRunner(cfg)
    resolver = VariableResolver()

    async def run_variant(template):
        results = []
        for tc in test_suite.test_cases:
            try:
                rendered = resolver.render(template, tc.input_variables)
                messages = build_messages(rendered)
                response = await runner.run_single(
                    messages, template.model_config, model
                )
                assertion = assertion_factory(
                    {"type": tc.assertion_type, "params": tc.assertion_params}
                )
                check = assertion.evaluate(response.content, tc.expected_output)
                results.append(TestResult(passed=check.passed, response=response))
            except Exception:
                results.append(TestResult(passed=False))
        return results

    results_a = await run_variant(prompt_a)
    results_b = await run_variant(prompt_b)
    return results_a, results_b


def _display_results(results: dict[str, list[TestResult]]) -> None:
    """Pretty-print evaluation results to the console."""
    calc = MetricsCalculator()
    all_metrics = {}

    for model, model_results in results.items():
        metrics = calc.calculate_all(model, model_results)
        all_metrics[model] = metrics

        table = Table(title=f"Results: {model}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Accuracy", f"{metrics.accuracy.accuracy:.1%}")
        table.add_row(
            "Pass / Fail",
            f"{metrics.accuracy.passed} / {metrics.accuracy.failed}",
        )
        table.add_row("Latency (p50)", f"{metrics.latency.p50:.0f}ms")
        table.add_row("Total Tokens", f"{metrics.tokens.total:,}")
        table.add_row("Total Cost", f"${metrics.cost.total_cost_usd:.4f}")
        console.print(table)
        console.print()

    if len(all_metrics) > 1:
        comparison = compare_models(all_metrics)
        console.print("[bold]Model Rankings:[/bold]")
        console.print(f"  Accuracy: {', '.join(comparison['accuracy_ranking'])}")
        console.print(f"  Cost:     {', '.join(comparison['cost_ranking'])}")
        console.print(f"  Latency:  {', '.join(comparison['latency_ranking'])}")


def _export_results(results: dict[str, list[TestResult]], path: Path) -> None:
    """Export results to a JSON file."""
    export = {}
    for model, model_results in results.items():
        export[model] = [
            {
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "content": r.response.content if r.response else None,
                "test_case_id": r.test_case.id if r.test_case else None,
            }
            for r in model_results
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export, indent=2))


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
