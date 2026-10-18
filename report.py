"""Create paper-ready figures and tables from core experiment logs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "fixed_sgd": "Fixed SGD",
    "adam": "Adam",
    "lamp_batch": "LAMP batch",
    "lamp_lr": "LAMP LR",
    "lamp_joint": "LAMP joint",
}
METHOD_ORDER = ["fixed_sgd", "adam", "lamp_batch", "lamp_lr", "lamp_joint"]
METHOD_COLORS = {
    "fixed_sgd": "#2c3e50",
    "adam": "#e67e22",
    "lamp_batch": "#16a085",
    "lamp_lr": "#2980b9",
    "lamp_joint": "#c0392b",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def discover_runs(input_root: Path) -> list[tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]]:
    runs = []
    for epochs_path in sorted(input_root.glob("runs/*/*/seed-*/epochs.csv")):
        steps_path = epochs_path.with_name("steps.csv")
        runs.append((epochs_path.parent, read_rows(epochs_path), read_rows(steps_path)))
    if not runs:
        raise ValueError(f"No experiment runs found under {input_root}")
    return runs


def target_reached(task: str, metric: float, targets: dict[str, float]) -> bool:
    if task == "clf_moons":
        return metric >= targets[task]
    return metric <= targets[task]


def auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    return float(np.sum(0.5 * (y_values[1:] + y_values[:-1]) * (x_values[1:] - x_values[:-1])))


def summarise_run(
    run_root: Path,
    epochs: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    targets: dict[str, float],
) -> dict[str, Any]:
    task = str(epochs[0]["task"])
    method = str(epochs[0]["method"])
    seed = int(epochs[0]["seed"])
    best = min(epochs, key=lambda row: row["val_metric"]) if task in {"reg_sine", "quadratic"} else max(
        epochs, key=lambda row: row["val_metric"]
    )
    reached = next((row for row in epochs if target_reached(task, row["val_metric"], targets)), None)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "best_val_metric": best["val_metric"],
        "final_test_metric": epochs[-1].get("test_metric", epochs[-1]["val_metric"]),
        "target": targets.get(task, ""),
        "target_reached": int(reached is not None),
        "time_to_target": reached["wall_time"] if reached else "",
        "examples_to_target": reached["examples_seen"] if reached else "",
        "aulc": auc([row["examples_seen"] for row in epochs], [row["val_loss"] for row in epochs]),
        "avg_batch": float(np.mean([row["batch"] for row in epochs])),
        "max_batch": max(row["batch"] for row in epochs),
        "final_lr": epochs[-1]["lr"],
        "chatter": int(sum(int(row.get("chatter", 0) or 0) for row in steps)),
        "run_root": str(run_root),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if key != "run_root"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fields})


def tex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_tex_table(rows: list[dict[str, Any]], columns: list[str], path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("\\begin{table}[t]\n\\centering\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n\\toprule\n")
        handle.write(" & ".join(tex_escape(column) for column in columns) + r" \\ \midrule" + "\n")
        for row in rows:
            values = [row.get(column, "") for column in columns]
            handle.write(" & ".join(tex_escape(value) for value in values) + r" \\" + "\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
        handle.write(f"\\caption{{{tex_escape(caption)}}}\n\\end{{table}}\n")


def save_figure(figure: plt.Figure, output_root: Path, name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_root / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output_root / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output_root / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def plot_schematic(output_root: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 2.2))
    axis.axis("off")
    boxes = [
        (0.02, "Batch $B_t$\nLR $\\eta_t$"),
        (0.23, "Gradient\nupdate"),
        (0.44, "Probe loss\nbefore / after"),
        (0.67, "Signed\ncontrol error"),
        (0.86, "Next $B$, $\\eta$"),
    ]
    for x, label in boxes:
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            transform=axis.transAxes,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eaf2f8", "edgecolor": "#2c3e50"},
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.06, 0.5),
            xytext=(left[0] + 0.06, 0.5),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#2c3e50"},
        )
    save_figure(figure, output_root, "fig-controller-schematic")


def plot_quadratic(summaries: list[dict[str, Any]], output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for summary in summaries:
        if summary["task"] != "quadratic":
            continue
        rows = read_rows(Path(summary["run_root"]) / "epochs.csv")
        label = METHOD_LABELS.get(summary["method"], summary["method"])
        axes[0].plot([row["epoch"] for row in rows], [row["val_loss"] for row in rows], label=label)
        axes[1].plot([row["epoch"] for row in rows], [row["batch"] for row in rows], label=label)
    axes[0].set(xlabel="Step", ylabel="Quadratic loss", yscale="log")
    axes[1].set(xlabel="Step", ylabel="Batch size")
    axes[0].legend(frameon=False)
    save_figure(figure, output_root, "fig-quadratic")


def plot_learning_curves(summaries: list[dict[str, Any]], output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)
    for axis, task in zip(axes, ("clf_moons", "reg_sine")):
        grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for summary in summaries:
            if summary["task"] == task:
                grouped[summary["method"]].append(read_rows(Path(summary["run_root"]) / "epochs.csv"))

        ordered_methods = [
            method
            for method in METHOD_ORDER
            if method in grouped
        ] + [method for method in grouped if method not in METHOD_ORDER]

        for method in ordered_methods:
            runs = grouped[method]
            x = np.asarray([row["examples_seen"] for row in runs[0]], dtype=float)
            y = np.asarray([[row["val_loss"] for row in run] for run in runs], dtype=float)
            mean = y.mean(axis=0)
            std = y.std(axis=0)
            color = METHOD_COLORS.get(method, "#2c3e50")
            axis.plot(x, mean, color=color, linewidth=2, label=METHOD_LABELS.get(method, method))
            if len(runs) > 1:
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        axis.set_xlabel("Examples seen")
        axis.set_ylabel("Validation loss")
        axis.set_title("Moons" if task == "clf_moons" else "Sine")
        axis.grid(alpha=0.15, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(figure, output_root, "fig-learning-curves")


def plot_method_comparison(summaries: list[dict[str, Any]], output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=False)
    for axis, task in zip(axes, ("clf_moons", "reg_sine")):
        grouped: dict[str, list[float]] = defaultdict(list)
        for summary in summaries:
            if summary["task"] == task:
                grouped[summary["method"]].append(float(summary["final_test_metric"]))
        if not grouped:
            continue
        label_to_key = {label: key for key, label in METHOD_LABELS.items()}
        ordered_methods = [
            method
            for method in METHOD_ORDER
            if method in grouped
        ] + [method for method in grouped if method not in METHOD_ORDER]
        labels = [METHOD_LABELS.get(method, method) for method in ordered_methods]
        values = [float(np.mean(grouped[method])) for method in ordered_methods]
        sems = [
            float(np.std(grouped[method], ddof=1) / np.sqrt(len(grouped[method]))) if len(grouped[method]) > 1 else 0.0
            for method in ordered_methods
        ]
        baseline = next((method for method in ordered_methods if method == "fixed_sgd"), None)
        colors = [METHOD_COLORS.get(method, "#2c3e50") for method in ordered_methods]
        axis.bar(labels, values, yerr=sems, color=colors, edgecolor="0.25", linewidth=0.8, capsize=4, error_kw={"elinewidth": 1.0})
        if baseline is not None:
            baseline_index = ordered_methods.index(baseline)
            axis.axhline(float(np.mean(grouped[baseline])), linestyle="--", color="0.5", linewidth=1.0, alpha=0.8)
            axis.annotate(
                "Fixed SGD baseline",
                xy=(baseline_index, float(np.mean(grouped[baseline]))),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
            )
        axis.set_xlabel("Method")
        axis.set_ylabel("Final test metric")
        axis.set_title("Moons" if task == "clf_moons" else "Sine")
        axis.grid(axis="y", alpha=0.15)
        axis.tick_params(axis="x", rotation=20)

    save_figure(figure, output_root, "fig-method-comparison")


def plot_controller(summaries: list[dict[str, Any]], output_root: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for summary in summaries:
        if summary["method"] != "lamp_joint" or summary["task"] == "quadratic":
            continue
        rows = read_rows(Path(summary["run_root"]) / "steps.csv")
        label = f"{summary['task']} seed {summary['seed']}"
        x = [row["examples_seen"] for row in rows]
        axes[0].plot(x, [row["batch"] for row in rows], label=label)
        axes[1].plot(x, [row["lr"] for row in rows])
        axes[2].plot(x, [row["error"] for row in rows])
    axes[0].set_ylabel("Batch")
    axes[1].set_ylabel("Learning rate")
    axes[2].set_ylabel("Control error")
    axes[2].set_xlabel("Examples seen")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(figure, output_root, "fig-controller-trajectories")


def generate_report(input_root: Path, output_root: Path, targets: dict[str, float]) -> None:
    runs = discover_runs(input_root)
    summaries = [summarise_run(root, epochs, steps, targets) for root, epochs, steps in runs]
    write_csv(summaries, output_root / "summary_by_run.csv")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary["task"], summary["method"])].append(summary)
    aggregate_rows = []
    for (task, method), rows in sorted(grouped.items()):
        metrics = [row["final_test_metric"] for row in rows]
        aggregate_rows.append(
            {
                "task": task,
                "method": METHOD_LABELS.get(method, method),
                "final_test_mean": float(np.mean(metrics)),
                "final_test_std": float(np.std(metrics)),
                "target_success_rate": float(np.mean([row["target_reached"] for row in rows])),
                "examples_to_target_mean": float(
                    np.mean([row["examples_to_target"] for row in rows if row["examples_to_target"] != ""])
                )
                if any(row["examples_to_target"] != "" for row in rows)
                else "",
                "aulc_mean": float(np.mean([row["aulc"] for row in rows])),
                "avg_batch_mean": float(np.mean([row["avg_batch"] for row in rows])),
                "max_batch_mean": float(np.mean([row["max_batch"] for row in rows])),
                "chatter_mean": float(np.mean([row["chatter"] for row in rows])),
            }
        )
    write_csv(aggregate_rows, output_root / "summary_by_method.csv")

    table_root = output_root / "tables"
    write_tex_table(
        aggregate_rows,
        ["task", "method", "final_test_mean", "final_test_std", "target_success_rate", "aulc_mean"],
        table_root / "table-main.tex",
        "Core LAMP results.",
    )
    write_tex_table(
        aggregate_rows,
        ["task", "method", "avg_batch_mean", "max_batch_mean", "chatter_mean"],
        table_root / "table-controller.tex",
        "Controller diagnostics.",
    )
    write_csv(aggregate_rows, table_root / "table-main.csv")
    write_csv(aggregate_rows, table_root / "table-controller.csv")

    figure_root = output_root / "figures"
    plot_schematic(figure_root)
    plot_quadratic(summaries, figure_root)
    plot_learning_curves(summaries, figure_root)
    plot_method_comparison(summaries, figure_root)
    plot_controller(summaries, figure_root)

    (output_root / "README.md").write_text(
        "# Core evidence package\n\n"
        "Figures are available as PDF, SVG, and 600 dpi PNG. "
        "Tables are available as LaTeX and CSV.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/core"))
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(
        args.input,
        args.output,
        {"clf_moons": 0.85, "reg_sine": 0.25, "quadratic": 0.02},
    )

