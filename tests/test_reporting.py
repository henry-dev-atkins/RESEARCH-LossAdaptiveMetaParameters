import csv
from pathlib import Path

from report import generate_report


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root: Path, task: str, method: str, seed: int) -> None:
    run_root = root / "runs" / task / method / f"seed-{seed}"
    rows = []
    for epoch in range(3):
        rows.append(
            {
                "seed": seed,
                "task": task,
                "method": method,
                "epoch": epoch,
                "examples_seen": (epoch + 1) * 10,
                "wall_time": float(epoch + 1),
                "train_loss": 1.0 / (epoch + 1),
                "train_metric": 0.5,
                "val_loss": 1.0 / (epoch + 1),
                "val_metric": 0.8 + 0.05 * epoch,
                "test_loss": 1.0 / (epoch + 1),
                "test_metric": 0.8 + 0.05 * epoch,
                "lr": 0.01,
                "batch": 64,
            }
        )
    write_rows(run_root / "epochs.csv", rows)
    write_rows(
        run_root / "steps.csv",
        [
            {
                "seed": seed,
                "task": task,
                "method": method,
                "epoch": 0,
                "step": 0,
                "examples_seen": 10,
                "wall_time": 1.0,
                "lr": 0.01,
                "batch": 64,
                "grad_norm2": 1.0,
                "delta_det": 0.1,
                "delta_obs": 0.05,
                "gap": 0.05,
                "gap_norm": 0.5,
                "raw_error": 0.3,
                "error": 0.3,
                "vB": 0.01,
                "chatter": 0,
            }
        ],
    )


def test_report_writes_publication_formats(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    for task in ("quadratic", "clf_moons", "reg_sine"):
        for method in ("fixed_sgd", "lamp_joint"):
            make_run(input_root, task, method, 1337)

    generate_report(
        input_root,
        output_root,
        {"clf_moons": 0.85, "reg_sine": 0.25, "quadratic": 0.02},
    )

    assert (output_root / "summary_by_run.csv").exists()
    assert (output_root / "tables" / "table-main.tex").exists()
    assert (output_root / "figures" / "fig-learning-curves.pdf").exists()
    assert (output_root / "figures" / "fig-learning-curves.svg").exists()
    assert (output_root / "figures" / "fig-learning-curves.png").exists()
    assert (output_root / "figures" / "fig-method-comparison.pdf").exists()
    assert (output_root / "figures" / "fig-method-comparison.svg").exists()
    assert (output_root / "figures" / "fig-method-comparison.png").exists()
