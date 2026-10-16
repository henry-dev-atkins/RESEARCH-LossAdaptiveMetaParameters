# Loss Adaptive Meta Parameters

This repository tests LOCCO, a feedback controller that adapts learning rate and
batch size from the predicted and observed loss decrease during optimisation.

The current controller uses a fixed probe batch. It computes a signed,
normalised error and applies smoothed log-space updates to the next step.

## Run the core study

The project uses `uv` and a CUDA-enabled PyTorch build.

```text
uv sync
uv run pytest -q
uv run python experiment.py --config configs/core.json --suite smoke --device cuda
uv run python experiment.py --config configs/core.json --suite core --device cuda
uv run python report.py --input results/core --output results/paper
```

The core study covers a noisy quadratic, moons classification, and sine
regression. It compares fixed SGD, Adam, batch-only LOCCO, learning-rate-only
LOCCO, and joint LOCCO over three seeds.

## Outputs

- Raw step and epoch logs are under `results/core/runs/`.
- Aggregated CSV summaries are under `results/paper/`.
- Figures are under `results/paper/figures/` as PDF, SVG, and 600 dpi PNG.
- Tables are under `results/paper/tables/` as LaTeX and CSV.

The generated evidence package is the current research result. The first
three-seed run favours Adam on both neural tasks. It does not support a claim
that LOCCO improves the current tasks.

The theoretical draft remains in `main.tex`. It contains unfinished proof and
claim placeholders and should not be treated as a completed paper.
