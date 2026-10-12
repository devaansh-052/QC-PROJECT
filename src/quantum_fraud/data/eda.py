"""Full-file descriptive EDA; no fitted transformations or model selection."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from quantum_fraud.data.paysim import load_paysim
from quantum_fraud.data.policy import BALANCE_COLUMNS

AMOUNT_EDGES = [0, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, np.inf]
AMOUNT_LABELS = ["0–<100", "100–<1k", "1k–<10k", "10k–<100k", "100k–<1m", "1m–<10m", "10m+"]


def rates_by(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Return denominators and fraud rates, including missing group values."""
    table = frame["isFraud"].groupby(groups, observed=True, dropna=False).agg(rows="size", fraud="sum")
    table["nonfraud"] = table["rows"] - table["fraud"]
    table["fraud_percentage"] = 100 * table["fraud"] / table["rows"]
    return table


def summarize_eda(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize an already validated frame without mutating/copying the full frame.

    Fixed amount bins are descriptive, not fitted preprocessing boundaries.
    Correlations use pairwise complete observations; constants return NaN.
    """
    tables = {
        "transaction_types": rates_by(frame, frame["type"]),
        "time_steps": rates_by(frame, frame["step"]),
        "amount_bands": rates_by(frame, pd.cut(frame["amount"], AMOUNT_EDGES, labels=AMOUNT_LABELS, right=False)),
        "existing_flag": rates_by(frame, frame["isFlaggedFraud"]),
    }
    amount_rows = []
    histograms = []
    valid_amount = frame["amount"].dropna()
    upper = max(1.0, float(np.log1p(valid_amount.max()))) if not valid_amount.empty else 1.0
    edges = np.linspace(0, upper, 41)
    for label in (0, 1):
        values = frame.loc[frame["isFraud"].eq(label), "amount"].dropna()
        summary = values.describe(percentiles=[.5, .9, .99]).to_dict()
        summary["isFraud"] = label
        amount_rows.append(summary)
        counts, _ = np.histogram(np.log1p(values), bins=edges)
        histograms.extend({"isFraud": label, "left": float(edges[i]), "right": float(edges[i + 1]), "count": int(count), "within_class_fraction": float(count / len(values)) if len(values) else 0.0} for i, count in enumerate(counts))
    tables["amount_summary"] = pd.DataFrame(amount_rows).set_index("isFraud")
    tables["amount_histogram"] = pd.DataFrame(histograms)
    balance_rows = []
    for column in BALANCE_COLUMNS:
        for label in (0, 1):
            values = frame.loc[frame["isFraud"].eq(label), column].dropna()
            balance_rows.append({"field": column, "isFraud": label, "observed": len(values), "zero_count": int(values.eq(0).sum()), "zero_percentage": float(values.eq(0).mean() * 100) if len(values) else np.nan, "median": float(values.median()) if len(values) else np.nan})
    tables["balance_diagnostics"] = pd.DataFrame(balance_rows)
    columns = ["step", "amount", "isFraud"]
    correlations = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for left in columns:
        for right in columns:
            valid = frame[left].notna() & frame[right].notna()
            x, y = frame.loc[valid, left], frame.loc[valid, right]
            correlations.loc[left, right] = x.corr(y) if len(x) > 1 and x.nunique() > 1 and y.nunique() > 1 else np.nan
    tables["primary_numeric_correlations"] = correlations
    return tables


def plot_eda(tables: dict[str, pd.DataFrame], validation: dict, output: Path) -> list[str]:
    """Render six aggregate-only figures using a headless canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output.mkdir(parents=True, exist_ok=True)
    saved = []

    def canvas(name: str, title: str, panels: int = 1):
        fig = Figure(figsize=(12, 4.8), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, panels, squeeze=False)[0]
        fig.suptitle(title)
        return fig, axes

    def save(fig, name: str):
        for extension in ("png", "pdf"):
            filename = f"{name}.{extension}"
            fig.savefig(output / filename, dpi=160)
            saved.append(filename)
        fig.clear()

    fig, (ax,) = canvas("class_balance", "How imbalanced is the target? Full-file counts")
    counts = [validation["target_distribution"][str(label)] for label in (0, 1)]
    ax.bar(["Nonfraud", "Fraud"], counts, color=["#246a9a", "#bb4430"])
    if min(counts) > 0:
        ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)" if min(counts) > 0 else "Transactions")
    for position, count in enumerate(counts):
        ax.text(position, count, f"{count:,}", ha="center", va="bottom")
    save(fig, "class_balance")

    fig, axes = canvas("transaction_types", "Which transaction types carry volume and fraud?", 2)
    table = tables["transaction_types"]
    labels = [str(value) for value in table.index]
    axes[0].bar(labels, table.rows, color="#246a9a")
    axes[0].set_ylabel("Transactions")
    axes[1].bar(labels, table.fraud_percentage, color="#bb4430")
    axes[1].set_ylabel("Fraud / all transactions of type (%)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, "transaction_types")

    fig, axes = canvas("amounts", "How do amounts differ, and where is fraud concentrated?", 2)
    hist = tables["amount_histogram"]
    for label, name, color in [(0, "Nonfraud", "#246a9a"), (1, "Fraud", "#bb4430")]:
        part = hist[hist.isFraud.eq(label)]
        axes[0].step((part.left + part.right) / 2, part.within_class_fraction, where="mid", label=name, color=color)
    axes[0].set_xlabel("ln(1 + amount in local currency)")
    axes[0].set_ylabel("Fraction within class per bin")
    axes[0].legend()
    table = tables["amount_bands"]
    axes[1].bar([str(value) for value in table.index], table.fraud_percentage, color="#bb4430")
    axes[1].set_ylabel("Fraud / transactions in amount band (%)")
    axes[1].tick_params(axis="x", labelrotation=35)
    save(fig, "amounts")

    fig, axes = canvas("time", "Does volume or fraud prevalence change over simulation time?", 2)
    table = tables["time_steps"].sort_index()
    axes[0].plot(table.index, table.rows, color="#246a9a")
    axes[0].set_ylabel("Transactions per observed step")
    axes[1].plot(table.index, table.fraud_percentage, color="#bb4430")
    axes[1].set_ylabel("Fraud / transactions in step (%)")
    for ax in axes:
        ax.set_xlabel("Simulation step (not wall-clock date)")
    save(fig, "time")

    fig, (ax,) = canvas("balances", "DIAGNOSTIC ONLY: zero balances by class — excluded from primary models")
    table = tables["balance_diagnostics"]
    x = np.arange(len(BALANCE_COLUMNS))
    for label, name, offset, color in [(0, "Nonfraud", -.18, "#246a9a"), (1, "Fraud", .18, "#bb4430")]:
        part = table[table.isFraud.eq(label)].set_index("field").reindex(BALANCE_COLUMNS)
        ax.bar(x + offset, part.zero_percentage, .36, label=name, color=color)
    ax.set_xticks(x, BALANCE_COLUMNS)
    ax.set_ylabel("Zero values / observed balances within class (%)")
    ax.legend()
    save(fig, "balance_diagnostics")

    fig, (ax,) = canvas("correlations", "Descriptive Pearson correlations — no causal or feature-selection claim")
    table = tables["primary_numeric_correlations"]
    im = ax.imshow(table.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(table)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for i in range(len(table)):
        for j in range(len(table)):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:.3f}" if pd.notna(value) else "undefined", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    save(fig, "primary_numeric_correlations")
    return saved


def run_eda(source: Path, results_dir: Path, seed: int = 42) -> Path:
    """Load once, aggregate, release full data, and write a unique completed run.

    Failed runs may leave partial artifacts but never a completed report.json.
    No sampling is used. Full-data EDA limits later claims of an untouched holdout.
    """
    source, results_dir = source.resolve(), results_dir.resolve()
    project_data = Path(__file__).resolve().parents[3] / "data"
    if results_dir.is_relative_to(project_data) or results_dir.is_relative_to(source.parent):
        raise ValueError("EDA outputs must be outside the dataset directory; use results/")
    frame, validation = load_paysim(source)
    tables = summarize_eda(frame)
    del frame
    run_id = datetime.now(timezone.utc).strftime("eda_%Y%m%dT%H%M%S_") + uuid4().hex[:12]
    metrics_dir = results_dir / "metrics" / run_id
    figures_dir = results_dir / "figures" / run_id
    metrics_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f"{name}.csv", index=True, float_format="%.17g")
    figures = plot_eda(tables, validation, figures_dir)
    report = {
        "status": "complete", "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full_dataset_descriptive_eda_not_model_evaluation",
        "validation": validation, "seed": seed, "sampling": "none",
        "versions": {name: version(name) for name in ("quantum-fraud", "numpy", "pandas", "matplotlib")},
        "tables": [f"{name}.csv" for name in tables],
        "figures_dir": str(figures_dir), "figures": figures,
        "limitations": ["Whole-file label inspection means a future split is not a completely untouched research holdout", "Balance fields are diagnostic only; primary feature policy is unchanged", "Rates in sparse groups are unstable; consult counts", "Correlation is pairwise Pearson, not causation; missing/constant values can be undefined"],
    }
    report_path = metrics_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path
