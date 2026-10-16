"""Reproducible core experiments for joint loss-driven control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class LOCCOConfig:
    """Configuration for the signed log-space controller."""

    gamma_tgt: float = 0.2
    rho: float = 0.9
    mu_B: float = 0.9
    deadband: float = 0.02
    batch_gain: float = 0.05
    learning_rate_gain: float = 0.01
    Ls: float = 1.0
    B_min: int = 8
    B_max: int = 512
    eta_min: float = 1e-4
    eta_max: float = 0.05
    eps: float = 1e-12
    enable_batch: bool = True
    enable_eta: bool = True


@dataclass(frozen=True)
class ControllerObservation:
    """Measurements taken around one optimiser update."""

    loss_before: float
    loss_after: float
    grad_norm2: float
    eta: float
    batch: int


@dataclass(frozen=True)
class ControllerDecision:
    """Next control values and diagnostics for one update."""

    eta: float
    batch: int
    delta_det: float
    delta_obs: float
    gap: float
    gap_norm: float
    raw_error: float
    error: float
    vB: float
    chatter: int


class LOCCOController:
    """Apply the signed, smoothed, log-space LOCCO update."""

    def __init__(self, config: LOCCOConfig, eta0: float, batch0: int) -> None:
        self.config = config
        self.vB = 0.0
        self.previous_error_sign = 0
        self.eta = self._clip_eta(eta0)
        self.batch = self._clip_batch(batch0)

    def _clip_eta(self, eta: float) -> float:
        return float(np.clip(eta, self.config.eta_min, self.config.eta_max))

    def _clip_batch(self, batch: int) -> int:
        return int(np.clip(round(batch), self.config.B_min, self.config.B_max))

    @staticmethod
    def _require_finite(values: Iterable[float]) -> None:
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("LOCCO received a non-finite observation")

    def update(self, observation: ControllerObservation) -> ControllerDecision:
        """Return the next controls from one measured optimiser update."""

        self._require_finite(
            (
                observation.loss_before,
                observation.loss_after,
                observation.grad_norm2,
                observation.eta,
                float(observation.batch),
            )
        )
        if observation.eta <= 0.0 or observation.batch <= 0:
            raise ValueError("LOCCO requires positive learning rate and batch size")
        if observation.grad_norm2 < 0.0:
            raise ValueError("LOCCO requires a non-negative squared gradient norm")

        delta_obs = observation.loss_before - observation.loss_after
        q = self.config.Ls * observation.grad_norm2
        delta_det = observation.eta * observation.grad_norm2 - 0.5 * observation.eta**2 * q
        gap = delta_det - delta_obs
        scale = observation.eta * max(observation.grad_norm2, self.config.eps)
        gap_norm = gap / scale
        raw_error = gap_norm - self.config.gamma_tgt
        error = 0.0 if abs(raw_error) < self.config.deadband else raw_error

        sign = int(math.copysign(1.0, error)) if error else 0
        chatter = int(sign and self.previous_error_sign and sign != self.previous_error_sign)
        if sign:
            self.previous_error_sign = sign

        if self.config.enable_batch:
            self.vB = self.config.mu_B * self.vB + (1.0 - self.config.mu_B) * self.config.batch_gain * error
            next_batch = self._clip_batch(math.exp(math.log(observation.batch) + self.vB))
        else:
            self.vB = 0.0
            next_batch = self._clip_batch(observation.batch)

        if self.config.enable_eta:
            next_eta = self._clip_eta(
                math.exp(math.log(observation.eta) - self.config.learning_rate_gain * error)
            )
        else:
            next_eta = self._clip_eta(observation.eta)

        self.eta = next_eta
        self.batch = next_batch
        return ControllerDecision(
            eta=next_eta,
            batch=next_batch,
            delta_det=delta_det,
            delta_obs=delta_obs,
            gap=gap,
            gap_norm=gap_norm,
            raw_error=raw_error,
            error=error,
            vB=self.vB,
            chatter=chatter,
        )


class MLP(nn.Module):
    """Small configurable multilayer perceptron for the synthetic tasks."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(depth):
            layers.extend((nn.Linear(current, width), nn.ReLU()))
            current = width
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    """Set all random sources used by the experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_moons(n_per_class: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.random(n_per_class) * math.pi
    first = np.c_[np.cos(t), np.sin(t)]
    second = np.c_[1.0 - np.cos(t) + 0.5, 1.0 - np.sin(t) - 0.2]
    features = np.vstack((first, second))
    labels = np.hstack((np.zeros(n_per_class), np.ones(n_per_class))).astype(np.int64)
    return features + rng.normal(scale=noise, size=features.shape), labels


def make_sine(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(n, 1))
    labels = np.sin(features) + 0.5 * np.cos(2.0 * features)
    labels += rng.normal(scale=noise, size=labels.shape)
    return features, labels.astype(np.float32)


def split_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    n_train = int(0.6 * len(features))
    n_val = int(0.2 * len(features))
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    return (features[train], labels[train]), (features[val], labels[val]), (features[test], labels[test])


def to_tensors(data: tuple[np.ndarray, np.ndarray]) -> TensorDataset:
    features, labels = data
    x = torch.as_tensor(features, dtype=torch.float32)
    if labels.ndim == 1:
        y = torch.as_tensor(labels, dtype=torch.long)
    else:
        y = torch.as_tensor(labels, dtype=torch.float32)
    return TensorDataset(x, y)


def make_task(task: str, seed: int) -> tuple[TensorDataset, TensorDataset, TensorDataset, str, int, int]:
    if task == "clf_moons":
        features, labels = make_moons(800, 0.25, seed)
        task_kind, out_dim = "clf", 2
    elif task == "reg_sine":
        features, labels = make_sine(2000, 0.2, seed)
        task_kind, out_dim = "reg", 1
    else:
        raise ValueError(f"Unknown task {task}")

    train, val, test = split_arrays(features, labels, seed)
    probe_size = max(32, int(0.1 * len(train[0])))
    probe = (train[0][:probe_size], train[1][:probe_size])
    optimiser_train = (train[0][probe_size:], train[1][probe_size:])
    return (
        to_tensors(optimiser_train),
        to_tensors(val),
        to_tensors(test),
        task_kind,
        features.shape[1],
        out_dim,
    )


def loss_and_metric(output: torch.Tensor, target: torch.Tensor, task_kind: str) -> tuple[torch.Tensor, float]:
    if task_kind == "clf":
        loss = F.cross_entropy(output, target)
        metric = float((output.argmax(dim=1) == target).float().mean().item())
    else:
        loss = F.mse_loss(output, target)
        metric = float(torch.sqrt(loss).item())
    return loss, metric


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset, task_kind: str, device: torch.device) -> tuple[float, float]:
    model.eval()
    features, labels = dataset.tensors
    output = model(features.to(device))
    loss, metric = loss_and_metric(output, labels.to(device), task_kind)
    return float(loss.item()), metric


def probe_loss(model: nn.Module, probe: TensorDataset, task_kind: str, device: torch.device) -> float:
    model.eval()
    features, labels = probe.tensors
    output = model(features.to(device))
    loss, _ = loss_and_metric(output, labels.to(device), task_kind)
    return float(loss.item())


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_neural_run(
    task: str,
    method: str,
    seed: int,
    device_name: str,
    config: dict[str, Any],
    output_root: Path,
    epochs_override: int | None = None,
) -> None:
    set_seed(seed)
    device = torch.device(device_name)
    train, val, test, task_kind, in_dim, out_dim = make_task(task, seed)
    probe = to_tensors((train.tensors[0][:32].numpy(), train.tensors[1][:32].numpy()))
    train = TensorDataset(train.tensors[0][32:], train.tensors[1][32:])
    model = MLP(in_dim, out_dim).to(device)
    eta0 = float(config["initial_learning_rate"])
    batch0 = int(config["initial_batch"])
    if method == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=eta0)
    else:
        optimiser = torch.optim.SGD(model.parameters(), lr=eta0)

    controller: LOCCOController | None = None
    if method.startswith("locco"):
        controller_config = LOCCOConfig(
            gamma_tgt=float(config["gamma_tgt"]),
            rho=float(config["rho"]),
            mu_B=float(config["mu_B"]),
            deadband=float(config["deadband"]),
            batch_gain=float(config["batch_gain"]),
            learning_rate_gain=float(config["learning_rate_gain"]),
            Ls=float(config["Ls"]),
            B_min=int(config["B_min"]),
            B_max=min(int(config["B_max"]), len(train)),
            eta_min=float(config["eta_min"]),
            eta_max=float(config["eta_max"]),
            enable_batch=method in {"locco_batch", "locco_joint"},
            enable_eta=method in {"locco_lr", "locco_joint"},
        )
        controller = LOCCOController(controller_config, eta0, batch0)

    eta, batch = eta0, batch0
    step_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    examples_seen = 0
    step_index = 0
    epochs = epochs_override or int(config["epochs"])
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train), generator=generator)
        train_loss_total = 0.0
        train_metric_total = 0.0
        train_count = 0
        for start in range(0, len(train), max(1, batch)):
            indices = order[start : start + max(1, batch)]
            features = train.tensors[0][indices].to(device)
            labels = train.tensors[1][indices].to(device)
            before_probe = probe_loss(model, probe, task_kind, device) if controller else 0.0
            optimiser.zero_grad(set_to_none=True)
            for group in optimiser.param_groups:
                group["lr"] = eta
            output = model(features)
            loss, metric = loss_and_metric(output, labels, task_kind)
            loss.backward()
            grad_norm2 = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_norm2 += float((parameter.grad.detach() ** 2).sum().item())
            optimiser.step()
            after_probe = probe_loss(model, probe, task_kind, device) if controller else before_probe

            decision = None
            if controller:
                decision = controller.update(
                    ControllerObservation(
                        loss_before=before_probe,
                        loss_after=after_probe,
                        grad_norm2=grad_norm2,
                        eta=eta,
                        batch=batch,
                    )
                )
                eta, batch = decision.eta, decision.batch

            examples_seen += len(indices)
            train_count += len(indices)
            train_loss_total += float(loss.item()) * len(indices)
            train_metric_total += metric * len(indices)
            step_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "method": method,
                    "epoch": epoch,
                    "step": step_index,
                    "examples_seen": examples_seen,
                    "wall_time": time.perf_counter() - start_time,
                    "lr": eta,
                    "batch": batch,
                    "grad_norm2": grad_norm2,
                    "delta_det": decision.delta_det if decision else "",
                    "delta_obs": decision.delta_obs if decision else "",
                    "gap": decision.gap if decision else "",
                    "gap_norm": decision.gap_norm if decision else "",
                    "raw_error": decision.raw_error if decision else "",
                    "error": decision.error if decision else "",
                    "vB": decision.vB if decision else "",
                    "chatter": decision.chatter if decision else 0,
                }
            )
            step_index += 1

        val_loss, val_metric = evaluate(model, val, task_kind, device)
        test_loss, test_metric = evaluate(model, test, task_kind, device)
        epoch_rows.append(
            {
                "seed": seed,
                "task": task,
                "method": method,
                "epoch": epoch,
                "examples_seen": examples_seen,
                "wall_time": time.perf_counter() - start_time,
                "train_loss": train_loss_total / train_count,
                "train_metric": train_metric_total / train_count,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "test_loss": test_loss,
                "test_metric": test_metric,
                "lr": eta,
                "batch": batch,
            }
        )

    run_root = output_root / "runs" / task / method / f"seed-{seed}"
    write_rows(step_rows, run_root / "steps.csv")
    write_rows(epoch_rows, run_root / "epochs.csv")


def run_quadratic(method: str, seed: int, config: dict[str, Any], output_root: Path, steps: int) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    theta = 3.0
    sigma = float(config["quadratic_sigma"])
    eta = float(config["quadratic_initial_learning_rate"])
    batch = int(config["quadratic_initial_batch"])
    controller = None
    if method == "locco_joint":
        controller = LOCCOController(
            LOCCOConfig(
                gamma_tgt=float(config["gamma_tgt"]),
                rho=float(config["rho"]),
                mu_B=float(config["mu_B"]),
                deadband=float(config["deadband"]),
                batch_gain=float(config["batch_gain"]),
                learning_rate_gain=float(config["learning_rate_gain"]),
                Ls=float(config["Ls"]),
                B_min=int(config["quadratic_B_min"]),
                B_max=int(config["quadratic_B_max"]),
                eta_min=float(config["eta_min"]),
                eta_max=float(config["eta_max"]),
            ),
            eta,
            batch,
        )
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        loss_before = 0.5 * theta**2
        grad = theta + sigma * float(rng.normal()) / math.sqrt(batch)
        grad_norm2 = theta**2
        theta -= eta * grad
        loss_after = 0.5 * theta**2
        decision = controller.update(
            ControllerObservation(loss_before, loss_after, grad_norm2, eta, batch)
        ) if controller else None
        if decision:
            eta, batch = decision.eta, decision.batch
        rows.append(
            {
                "seed": seed,
                "task": "quadratic",
                "method": method,
                "epoch": step,
                "step": step,
                "examples_seen": (step + 1) * batch,
                "wall_time": "",
                "train_loss": loss_after,
                "val_loss": loss_after,
                "test_loss": loss_after,
                "val_metric": loss_after,
                "lr": eta,
                "batch": batch,
                "theta": theta,
                "delta_det": decision.delta_det if decision else "",
                "delta_obs": decision.delta_obs if decision else "",
                "gap": decision.gap if decision else "",
                "gap_norm": decision.gap_norm if decision else "",
                "raw_error": decision.raw_error if decision else "",
                "error": decision.error if decision else "",
                "vB": decision.vB if decision else "",
                "chatter": decision.chatter if decision else 0,
            }
        )
    run_root = output_root / "runs" / "quadratic" / method / f"seed-{seed}"
    write_rows(rows, run_root / "steps.csv")
    write_rows(rows, run_root / "epochs.csv")


def run_suite(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.suite == "smoke":
        run_quadratic("locco_joint", args.seed, config, output_root, steps=8)
        train_neural_run("clf_moons", "locco_joint", args.seed, args.device, config, output_root, epochs_override=2)
        return

    seeds = [int(value) for value in config["seeds"]]
    for seed in seeds:
        for method in config["methods"]:
            if method == "adam" or method.startswith("locco") or method == "fixed_sgd":
                for task in ("clf_moons", "reg_sine"):
                    train_neural_run(task, method, seed, args.device, config, output_root)
        for method in ("fixed_sgd", "locco_joint"):
            run_quadratic(method, seed, config, output_root, steps=int(config["quadratic_steps"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/core.json")
    parser.add_argument("--suite", choices=("smoke", "core"), default="smoke")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/core")
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())

