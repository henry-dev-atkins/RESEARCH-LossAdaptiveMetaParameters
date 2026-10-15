# PRD: Loss-Driven Online Control of Learning Rate and Batch Size (LOCCO) — Experiment Suite
**File:** PRD.md  
**Version:** 1.0  
**Owner:** (You)  
**Primary Engineer:** (Assignee)  
**Stack:** Python 3.10, PyTorch >= 2.0, matplotlib, NumPy

---

## 0) Executive Summary

We will build a **reproducible PyTorch experiment suite** that evaluates a loss-driven controller (LOCCO) which adapts **batch size** (and optionally **learning rate**) online based on **observed loss reduction**. The suite must produce **publish-standard results** across proof-of-concept (POC), benchmark, ablation, and robustness experiments.

Outputs include **CSV logs**, **PNG plots**, and a **summary CSV** ready to drop into a paper. The code must be deterministic where feasible and have a clear CLI.

---

## 1) Problem Statement & Goals

### 1.1 Problem
Choosing the **learning rate (LR)** and **batch size (BS)** is brittle, workload-specific, and costly to tune. We propose a **feedback controller** that uses the **loss-gap** (difference between predicted deterministic loss decrease and observed decrease) to **co-adapt** LR and BS during training.

### 1.2 Goals
- Implement **LOCCO** controller in PyTorch training loops.
- Compare LOCCO against baselines (SGD, cosine-SGD, Adam/AdamW) on **classification** and **regression** tasks, with both **deep** and **conventional** models.
- Produce **POC, benchmark, ablation, and robustness** results.
- Emit **plots and tables** suitable for a paper (time-to-target, examples-to-target, AULC, final metrics).
- Ensure **reproducibility** (seeds, deterministic flags) and **clear configuration**.

### 1.3 Non-goals
- Achieving SOTA accuracy on large-scale datasets. The focus is **efficiency and stability**, not absolute SOTA.

---

## 2) Definitions & Concepts

- **Loss-gap:** \(\text{gap}_t = \Delta L_{\mathrm{det},t} - \Delta L_{\mathrm{obs},t}\).  
- **Deterministic drop (proxy):** \(\Delta L_{\mathrm{det},t} \approx \eta_t \|\widehat g_t\|^2 - \frac{\eta_t^2}{2}\,q_t\), with \(q_t \approx \widehat g_t^\top H_t \widehat g_t\) (or a capped proxy \(L_s\|\widehat g_t\|^2\)).
- **Observed drop:** \(\Delta L_{\mathrm{obs},t}= L(\theta_t)-L(\theta_{t+1})\). Prefer a **probe batch** to evaluate.
- **Variance budget (intuition):** stochastic penalty scales like \(\eta_t^2/B_t\). LOCCO targets a **dimensionless** fraction \(\tilde\gamma\) of deterministic decrease.
- **Signed control error:** \(e_t = \frac{\text{gap}_t}{\eta_t\|\widehat g_t\|^2+\varepsilon} - \tilde\gamma\).

---

## 3) Functional Requirements

### 3.1 Inputs
- **Datasets (synthetic):**
  - Classification: Moons (2D), Blobs (multi-class, d≈10–12).
  - Regression: Sine (1D), Linear(d≈20–25).
- **Models:**
  - Conventional: Linear regression/classifier.
  - Deep: MLP (configurable width/depth).
- **Optimizers:** SGD (±momentum), cosine-SGD, Adam/AdamW.
- **Controller Config (LOCCO):**
  - `use` (bool), `alpha` (batch gain), `beta` (LR gain), `rho` (EMA loss smoothing), `Ls` (smoothness cap), `use_curv_proxy` (bool), `preserve_eta_over_B` (bool), `B_min`, `B_max`, `gamma_tgt`, `deadband`, `hysteresis_k`, etc.
- **Run Config:**
  - `task` (“clf” or “reg”), `epochs`, `batch`, `lr`, `weight_decay`, seeds.
  - Ablation flags: smoothing off, curvature off, fixed batch, signed/rectified error.

### 3.2 Outputs
- **Per-run CSV:** epochwise metrics: `epoch, lr, batch, train_loss, val_loss, test_loss, train_acc|train_rmse, val_acc|val_rmse, test_acc|test_rmse`.
- **Per-run PNG:** learning curves with secondary y-axis for `lr` and `batch` (and optional loss-gap track).
- **Summary CSV:** one row per run: best validation metric, final test metric, time-to-target, examples-to-target, avg/max batch, final lr.
- **Saved models** (optional): last/best checkpoints for reproducibility.

### 3.3 Non-Functional
- Determinism: fixed seeds, cudnn deterministic, reproducible data splits.
- Performance: reasonable runtime on CPU/GPU for the synthetic suite.
- Code quality: modular, documented, typed where practical.

---

## 4) High-Level Architecture

experiment.py
├── data.py (synthetic generators, splits, loaders)
├── models.py (LinearModel, MLP)
├── locco.py (controller class + update logic)
├── train.py (train_one_epoch, evaluate, metrics)
├── runners.py (suite orchestration: POC/Bench/Abla/Robust)
├── viz.py (plotting)
├── utils.py (seeding, logging, csv io, config parsing)
└── results/ (CSV, PNG, SUMMARY.csv, checkpoints/)

markdown
Always show details

Copy code

The **runner** composes dataset → model → optimizer → controller → training loop, logs metrics, and generates plots/tables.

---

## 5) Detailed Specifications

### 5.1 Data

#### 5.1.1 Synthetic Classification
- **Moons**: `n=800`, noise σ≈0.2–0.3. Labels possibly flipped at rate `p` in robustness runs.
- **Blobs**: `n=3000`, `k=7` classes, `dim=12`, std≈1.5–1.8.

#### 5.1.2 Synthetic Regression
- **Sine**: `y = sin(x) + 0.5 cos(2x) + N(0,σ^2)`; `n=2000`, `x∈[-3,3]`.
- **Linear**: `y = X w + N(0,σ^2)` with `dim=25`, `n=2000`.

#### 5.1.3 Splits
- Train/Val/Test = 60/20/20 (stratified for classification). Fixed seed shuffling.

### 5.2 Models
- **LinearModel(in_dim, out_dim)**: single linear layer.
- **MLP(in_dim, out_dim, width, depth, act=ReLU)**: `depth` × (Linear → Act) + output layer. Optional BN is off by default for simplicity.

### 5.3 Optimizers
- **SGD**: lr, weight_decay, optional momentum/Nesterov.
- **SGD+Cosine**: cosine annealing scheduler over `epochs`.
- **Adam/AdamW**: default betas; WD per model/task.

### 5.4 LOCCO Controller

#### 5.4.1 Signals
- `grad_norm2 = Σ ||∂L/∂θ||²` from current step.
- Deterministic proxy: `delta_det = eta * grad_norm2 - 0.5 * eta^2 * q`, with `q = Ls * grad_norm2` (cap), or **HVP** when enabled.
- Observed drop (prefer **probe batch**): `delta_obs = L(θ_t) - L_probe(θ_{t+1})`. If probe not available, use EMA of training loss: `barL_{t+1} = ρ barL_t + (1-ρ) L_train(θ_{t+1})` and `delta_obs = L_train(θ_t) - barL_{t+1}`.

#### 5.4.2 Control Error (signed, dimensionless)
- `s_t = eta * max(grad_norm2, eps)`
- `gap = delta_det - delta_obs`
- `err = (gap / s_t) - gamma_tgt`

#### 5.4.3 Update (log-space with deadband + hysteresis)
- If `|err| < deadband`: set `err = 0`.
- Maintain a low-pass state `vB = μB * vB + (1-μB) * kB * err` (μB≈0.9).
- `logB_new = log(B_t) + vB`; `B_{t+1} = clip(round(exp(logB_new)), B_min, B_max)`.
- (Optional) LR guard: preserve `eta/B` on **decrease** of `B` only; otherwise LR follows schedule or separate LOCCO `beta` rule.

**Rationale:**  
- Signed error allows \(B\) to **increase or decrease**.  
- Log-space + deadband reduces **chatter**.  
- HVP reduces curvature overestimation; if disabled, `Ls` cap prevents over-shrinking `delta_det`.

### 5.5 Training Loop
- Manual micro-iteration to support per-step `B` change when LOCCO is active.
- After each optimizer step: compute controller signals and update `(eta, B)` for **next** step.
- LR per-step override via `param_group["lr"]` if needed.

### 5.6 Metrics

#### 5.6.1 Core
- **Classification:** accuracy (val/test), cross-entropy loss (train/val/test).
- **Regression:** RMSE (val/test), MSE loss.

#### 5.6.2 Efficiency
- **Time-to-target:** first epoch/time reaching a preset loss/accuracy threshold per task.
- **Examples-to-target:** cumulative samples processed to reach target.
- **AULC:** area under loss curve up to fixed wall time/epoch budget.
- **Throughput:** samples/sec; peak memory (optional).

#### 5.6.3 Controller Diagnostics
- Trajectories of `B_t`, `eta_t`.
- `gap_norm = gap / s_t` to show tracking around `gamma_tgt`.
- Histogram of `B_t`; # of **chatter** events (sign flips beyond deadband).

### 5.7 Logging & Visualization
- CSV per run (epochwise), PNG plots: loss curves + `lr`/`batch` on secondary axis.
- Summary CSV (best val metric, final test, time/examples-to-target, avg/max B).

### 5.8 CLI
python experiment.py --all # full suite
python experiment.py # quick POC subset
--seed 1337 --batch 64 # global defaults

yaml
Always show details

Copy code
Add flags if needed: `--probe-eval`, `--hvp`, `--deadband 0.02`, etc.

---


## 6) Experimental Design

### 6.1 POC
- **Moons (clf)** + **Sine (reg)** with Linear and MLP.
- Methods: SGD, Adam, SGD+cosine, LOCCO.
- Plots: loss vs epoch/examples/time; `B_t` & `eta_t` vs epoch; `gap_norm` vs epoch.
- Table: final metric, time/examples-to-target, AULC, avg/max B.

### 6.2 Benchmarks
- **Blobs (clf, d=12, k=7, MLP 4×256)** and **Linear (reg, d=25, MLP 4×256)**.
- Methods: SGD+cosine, Adam, LOCCO.
- Same plots/tables as POC; include throughput and (optional) peak mem.

### 6.3 Ablations
- LOCCO-noSmooth (ρ=0), LOCCO-noCurv (no HVP/low Ls), FixedB (no control), Rectified error (to show monotone \(B\)).
- Table: final metric, time-to-target, AULC, var(B), var(eta), chatter events.

### 6.4 Robustness/Extrema
- LR ∈ {1e-4, 1.0}, B ∈ {8, 2048}, label noise 30%, sampling w/o replacement (if added).
- Heatmaps over (init LR, init B) of time-to-target; success/failure map.
- Claim: LOCCO enlarges feasible hyperparameter region and degrades gracefully.

---


## 7) Analysis Considerations

- **Measurement bias:** Prefer probe batch for `delta_obs`. If unavailable, EMA introduces lag; track and report `rho`.
- **Curvature proxy:** HVP is preferred; otherwise cap `Ls`. Overestimation biases toward larger \(B\); acknowledge in discussion.
- **Edge cases:** when \(\|g\|^2\) is tiny, reduce \(\eta\) rather than pushing \(B\rightarrow B_{\max}\).
- **Statistical reporting:** run **3 seeds** for benchmark results; report mean±std.
- **Compute accounting:** report wall time, examples, and epochs.

---


## 8) Acceptance Criteria (Definition of Done)

1. **Reproducibility:** reruns with same seeds reproduce summary metrics within ±1% relative.
2. **Artifacts:** For each run, CSV and PNG exist; `SUMMARY.csv` aggregates best-val and final test metrics.
3. **POC evidence:** LOCCO improves time-to-target vs at least one baseline on both clf and reg.
4. **Ablations:** show monotone \(B\) when error is rectified and **non-monotone** \(B\) with signed/log-space control.
5. **Robustness:** LOCCO succeeds (reaches target) on at least **two** extreme settings where a baseline fails.
6. **Documentation:** README with how-to-run and how-to-interpret.

---


## 9) Risks & Mitigations

- **Noisy control signal** → use probe batch, EMA smoothing, deadband, hysteresis.
- **Curvature misestimation** → prefer HVP; otherwise cap with running \(L_s\).
- **Chatter** → log-space updates, deadband, low-pass filter (μB≈0.9).
- **Generalization under large \(B\)** → monitor train-test gap; cap \(B_{\max}\); consider mixup/weight decay if needed.

---


## 10) Implementation Plan (Engineering Tasks)

1. Scaffold modules (`data.py, models.py, locco.py, train.py, runners.py, viz.py`).
2. Implement datasets + loaders, splits with seeds.
3. Implement models (Linear, MLP).
4. Implement optimizers + schedulers.
5. Implement LOCCO (signed error, log-space, deadband, hysteresis, optional HVP).
6. Training loop with per-step override of `(eta, B)`.
7. Evaluation, logging (CSV), plotting.
8. Runners for POC, Benchmarks, Ablations, Robustness.
9. CLI + README.
10. QA: deterministic reruns, unit tests for controller math (impulse tests), smoke tests.

---


## 11) Appendix: Pseudocode

### 11.1 LOCCO (Batch-First Variant)
Inputs: eta_t, B_t, grad_norm2, L_t, L_tp1, Ls, rho, gamma_tgt, deadband, muB, kB
State: ema_loss, vB

ema_loss = rho * ema_loss + (1 - rho) * L_tp1
delta_obs = max(0, L_t - ema_loss) # or probe loss
q = min(Ls_cap, Ls_running_avg) * grad_norm2 # or HVP
delta_det = eta_t * grad_norm2 - 0.5 * eta_t^2 * q
gap = delta_det - delta_obs
s_t = eta_t * max(grad_norm2, eps)
err = (gap / s_t) - gamma_tgt
if |err| < deadband: err = 0
vB = muB * vB + (1 - muB) * kB * err
logB_new = log(B_t) + vB
B_{t+1} = clip(round(exp(logB_new)), B_min, B_max)
(Optional) Adjust eta_{t+1} separately or preserve eta/B on decreases only.

shell
Always show details

Copy code

### 11.2 Training Loop (skeletal)
for epoch in 0..E-1:
shuffle training data deterministically
idx = 0
while idx < N_train:
B_step = B_t
batch = data[idx : idx + B_step]
loss, grad_norm2 = forward_backward_step(batch, eta_t)
optimizer.step()
# Controller
eta_{t+1}, B_{t+1} = LOCCO.update(loss_t, loss_tp1, grad_norm2, eta_t, B_t)
# Logging
record epoch metrics and controller diagnostics

yaml
Always show details

Copy code

---


## 12) Deliverables

- Code repo with modules and `experiment.py` CLI.
- `results/` with per-run CSVs and PNGs, plus `SUMMARY.csv`.
- README: instructions + mapping from outputs to paper figures/tables.
- This **PRD.md**.
