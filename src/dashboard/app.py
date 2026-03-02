"""Streamlit dashboard for prompt evaluation visualization.

Displays evaluation metrics, model comparison, latency analysis,
cost breakdown, and A/B test results using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def generate_eval_results(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic evaluation results per model."""
    rng = np.random.default_rng(seed)
    models = ["gpt-4o", "claude-3-5-sonnet", "gpt-4o-mini", "claude-3-haiku"]
    rows = []
    for model in models:
        total = int(rng.integers(80, 150))
        passed = int(total * rng.uniform(0.65, 0.95))
        rows.append(
            {
                "model": model,
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy": round(passed / total, 4),
                "avg_latency_ms": round(rng.uniform(200, 3000), 1),
                "p95_latency_ms": round(rng.uniform(500, 8000), 1),
            }
        )
    return pd.DataFrame(rows)


def generate_token_usage(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic token usage data per model."""
    rng = np.random.default_rng(seed)
    models = ["gpt-4o", "claude-3-5-sonnet", "gpt-4o-mini", "claude-3-haiku"]
    rows = []
    for model in models:
        input_tok = int(rng.integers(50000, 300000))
        output_tok = int(rng.integers(20000, 100000))
        rate = {
            "gpt-4o": 5.0,
            "claude-3-5-sonnet": 3.0,
            "gpt-4o-mini": 0.15,
            "claude-3-haiku": 0.25,
        }[model]
        cost = (input_tok + output_tok) * rate / 1_000_000
        rows.append(
            {
                "model": model,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "total_cost": round(cost, 4),
                "cost_per_test": round(cost / int(rng.integers(80, 150)), 6),
            }
        )
    return pd.DataFrame(rows)


def generate_accuracy_by_tag(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic accuracy by test tag."""
    rng = np.random.default_rng(seed)
    tags = ["reasoning", "math", "coding", "creative", "factual", "summarization"]
    rows = []
    for tag in tags:
        rows.append(
            {
                "tag": tag,
                "accuracy": round(rng.uniform(0.55, 0.95), 4),
                "test_count": int(rng.integers(15, 50)),
            }
        )
    return pd.DataFrame(rows)


def generate_ab_test_results(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic A/B test comparison data."""
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "avg_latency_ms", "cost_per_test", "consistency"]
    rows = []
    for metric in metrics:
        base = {
            "accuracy": 0.82,
            "avg_latency_ms": 1200.0,
            "cost_per_test": 0.015,
            "consistency": 0.78,
        }[metric]
        rows.append(
            {
                "metric": metric,
                "variant_a": round(base + rng.uniform(-0.05, 0.05) * base, 4),
                "variant_b": round(base + rng.uniform(-0.05, 0.05) * base, 4),
                "p_value": round(rng.uniform(0.001, 0.2), 4),
            }
        )
    return pd.DataFrame(rows)


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Prompt Evaluation Framework Dashboard")
    st.caption(
        "Multi-model prompt testing with accuracy metrics, latency analysis, "
        "cost optimization, and A/B test comparison"
    )


def render_summary_metrics(eval_df: pd.DataFrame, token_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tests", eval_df["total_tests"].sum())
    best = eval_df.loc[eval_df["accuracy"].idxmax()]
    col2.metric("Best Accuracy", f"{best['accuracy']:.1%}")
    col3.metric("Models Tested", len(eval_df))
    col4.metric("Total Cost", f"${token_df['total_cost'].sum():.2f}")


def render_model_accuracy(eval_df: pd.DataFrame) -> None:
    """Render model accuracy comparison."""
    st.subheader("Model Accuracy Comparison")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=eval_df["model"],
            y=eval_df["accuracy"],
            marker_color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
            text=eval_df["accuracy"].apply(lambda x: f"{x:.1%}"),
            textposition="auto",
        )
    )
    fig.update_layout(
        yaxis={"range": [0.5, 1.0], "tickformat": ".0%"},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latency_chart(eval_df: pd.DataFrame) -> None:
    """Render latency comparison chart."""
    st.subheader("Latency Analysis")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Avg Latency",
            x=eval_df["model"],
            y=eval_df["avg_latency_ms"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="P95 Latency",
            x=eval_df["model"],
            y=eval_df["p95_latency_ms"],
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Latency (ms)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown(token_df: pd.DataFrame) -> None:
    """Render cost breakdown per model."""
    st.subheader("Cost Analysis")
    fig = px.pie(
        token_df,
        values="total_cost",
        names="model",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_accuracy_by_tag(tag_df: pd.DataFrame) -> None:
    """Render accuracy by test category."""
    st.subheader("Accuracy by Category")
    fig = px.bar(
        tag_df.sort_values("accuracy"),
        x="accuracy",
        y="tag",
        orientation="h",
        color="accuracy",
        color_continuous_scale="RdYlGn",
        text="accuracy",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="auto")
    fig.update_layout(
        xaxis={"tickformat": ".0%"},
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ab_results(ab_df: pd.DataFrame) -> None:
    """Render A/B test comparison table."""
    st.subheader("A/B Test Results")
    st.dataframe(ab_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    eval_df = generate_eval_results()
    token_df = generate_token_usage()
    tag_df = generate_accuracy_by_tag()
    ab_df = generate_ab_test_results()

    render_summary_metrics(eval_df, token_df)
    st.markdown("---")

    render_model_accuracy(eval_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_latency_chart(eval_df)
    with col_right:
        render_cost_breakdown(token_df)

    st.markdown("---")
    render_accuracy_by_tag(tag_df)
    render_ab_results(ab_df)


if __name__ == "__main__":
    main()
