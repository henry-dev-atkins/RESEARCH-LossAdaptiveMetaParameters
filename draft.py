
# experiment.py
# End-to-end PyTorch experiments: POC + Benchmarks + Ablations + Extrema
# Author: you
#
# Set this variable to control the run mode:
# RUN_TYPE = "POC" or "ALL" or any other custom string you want to use in code
RUN_TYPE = "ALL"  # Change this as needed
RUN_SEED = 1337    # Change this as needed

import os, math, random, time, csv
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------- Repro ----------------
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------- Datasets (Synthetic) ----------------
def make_moons(n=400, noise=0.2, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.random(n) * math.pi
    x1 = np.c_[np.cos(t), np.sin(t)]
    x2 = np.c_[1 - np.cos(t) + 0.5, 1 - np.sin(t) - 0.2]
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int64)
    X += rng.normal(scale=noise, size=X.shape)
    return X, y

def make_blobs(n=2000, k=5, dim=10, std=1.5, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(k, dim))
    ns = [n//k]*(k-1)+[n - (k-1)*(n//k)]
    Xs, ys = [], []
    for c, m, i in zip(centers, ns, range(k)):
        Xs.append(rng.normal(loc=c, scale=std, size=(m, dim)))
        ys.append(np.full(m, i, dtype=np.int64))
    X, y = np.vstack(Xs), np.hstack(ys)
    return X, y

def make_regression_sine(n=1000, noise=0.2, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X) + 0.5*np.cos(2*X) + rng.normal(scale=noise, size=(n, 1))
    return X, y

def make_regression_linear(n=1000, dim=20, noise=0.5, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(dim, 1))
    X = rng.normal(size=(n, dim))
    y = X @ W + rng.normal(scale=noise, size=(n, 1))
    return X, y

# -------------- Models ----------------
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=3, act=nn.ReLU, bn=False):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width)]
            if bn: layers += [nn.BatchNorm1d(width)]
            layers += [act()]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# -------------- Metrics ----------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

def rmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(pred, y)).item()

# -------------- Controller (LOCCO) ----------------
@dataclass
class LoccoConfig:
    use: bool = False
    alpha: float = 0.05        # batch-size gain
    beta: float = 0.00         # (optional) LR gain (0 => LR fixed)
    rho: float = 0.9           # EMA loss smoothing
    eps: float = 1e-12
    Ls: float = 1.0            # smoothness upper bound (curvature proxy)
    use_curv_proxy: bool = False # if True, attempt curvature proxy via Ls*||g||^2
    preserve_eta_over_B: bool = False # keep eta/B invariant if True
    B_min: int = 8
    B_max: int = 4096
    min_lr: float = 1e-5   # minimum allowed learning rate
    max_lr: float = 0.05   # maximum allowed learning rate

class LoccoState:
    def __init__(self, cfg: LoccoConfig, eta0: float, B0: int, loss0: float):
        self.cfg = cfg
        self.ema_loss = loss0
        self.prev_eta = eta0
        self.prev_B = B0

    def step(self, t: int, loss_t: float, loss_tp1: float,
             grad_norm2: float, eta_t: float, B_t: int) -> Tuple[float, int]:
        if not self.cfg.use:
            return eta_t, B_t
        # Smooth loss
        self.ema_loss = self.cfg.rho * self.ema_loss + (1 - self.cfg.rho) * loss_tp1
        delta_obs = loss_t - self.ema_loss
        # Deterministic proxy
        if self.cfg.use_curv_proxy:
            q = self.cfg.Ls * grad_norm2  # safe upper bound of g^T H g
        else:
            q = self.cfg.Ls * grad_norm2  # keep simple & safe
        delta_det = eta_t * grad_norm2 - 0.5 * (eta_t ** 2) * q
        delta_stoch_hat = delta_det - delta_obs
        s_t = eta_t * max(grad_norm2, self.cfg.eps)


        # Adapt B with robust check
        B_float = B_t * (1.0 + self.cfg.alpha * (delta_stoch_hat / max(s_t, self.cfg.eps)))
        if not np.isfinite(B_float) or np.isnan(B_float):
            # Fallback to previous batch size or minimum
            B_new = max(self.cfg.B_min, min(B_t, self.cfg.B_max))
        else:
            B_new = int(np.clip(B_float, self.cfg.B_min, self.cfg.B_max))
        if B_new % 2 == 1:  # prefer even for better sharding
            B_new = min(self.cfg.B_max, B_new + 1)

        # Adapt eta (optional direct, or preserve eta/B)
        if self.cfg.preserve_eta_over_B and B_t > 0:
            eta_new = min(eta_t * (B_new / B_t), self.prev_eta * (self.cfg.B_max / self.cfg.B_min))
        else:
            eta_new = eta_t * (1.0 + self.cfg.beta * ((delta_det - delta_stoch_hat) / max(s_t, self.cfg.eps)))
        # Clamp learning rate to safe range
        eta_new = float(np.clip(eta_new, self.cfg.min_lr, self.cfg.max_lr))
        self.prev_eta, self.prev_B = eta_new, B_new
        return float(eta_new), int(B_new)

# -------------- Training boilerplate ----------------
def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle=True) -> DataLoader:
    tensX = torch.tensor(X, dtype=torch.float32)
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1 and y.dtype.kind in "iu"):
        tensy = torch.tensor(y.squeeze(), dtype=torch.long if y.dtype.kind in "iu" else torch.float32)
    else:
        tensy = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tensX, tensy), batch_size=batch_size, shuffle=shuffle, drop_last=False)

from typing import Optional

def train_one_epoch(model, loader, task, optimizer, device, locco_state: Optional[LoccoState]=None,
                    eta: Optional[float]=None, B: Optional[int]=None, L_s: float=1.0):
    model.train()
    total_loss, total_n = 0.0, 0
    correct = 0
    # Manual batching to allow per-step B changes, if LOCCO is used
    X_all, y_all = next(iter(DataLoader(loader.dataset, batch_size=len(loader.dataset), shuffle=True)))
    X_all, y_all = X_all.to(device), y_all.to(device)
    idx = 0
    while idx < len(X_all):
        B_step = B if (locco_state and locco_state.cfg.use) else loader.batch_size
        if B_step is None:
            raise ValueError(f"Batch size must be specified if LOCCO is not used. B_step is {B_step}")
        B_step = max(1, int(B_step))
        xb = X_all[idx: idx + B_step]
        yb = y_all[idx: idx + B_step]
        idx += B_step

        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        if task == "clf":
            loss = F.cross_entropy(out, yb)
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
        else:
            loss = F.mse_loss(out, yb)
            total_loss += loss.item() * len(xb)
        loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # Override LR per-step (SGD/Adam accept param_group lr changes)
        if eta is not None:
            for g in optimizer.param_groups:
                g["lr"] = eta
        optimizer.step()

        # After stepping, LOCCO adjustment (uses current batch signal)
        if locco_state and locco_state.cfg.use:
            with torch.no_grad():
                grad_norm2 = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm2 += (p.grad.detach()**2).sum().item()
            # Approximate losses around this micro-step:
            loss_t = total_loss / max(1, total_n + len(xb))  # running avg proxy
            loss_tp1 = loss.item()
            if eta is None:
                raise ValueError("eta must be provided if LOCCO is used")
            if B is None:
                raise ValueError("B must be provided if LOCCO is used")
            eta_next, B_next = locco_state.step(
                t=0, loss_t=float(loss_t), loss_tp1=float(loss_tp1),
                grad_norm2=float(grad_norm2), eta_t=float(eta), B_t=int(B_step)
            )
            eta, B = eta_next, B_next

        total_n += len(xb)

    avg_loss = total_loss / total_n
    if task == "clf":
        acc = correct / total_n
    else:
        acc = None
    return avg_loss, acc, eta, B

@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if task == "clf":
            loss = F.cross_entropy(out, yb)
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
        else:
            loss = F.mse_loss(out, yb)
            total_loss += loss.item() * len(xb)
    avg_loss = total_loss / len(loader.dataset)
    if task == "clf":
        acc = correct / len(loader.dataset)
        return avg_loss, acc
    else:
        return avg_loss, None

# -------------- Experiment harness ----------------
@dataclass
class RunConfig:
    name: str
    task: str                  # "clf" or "reg"
    model_type: str            # "linear" or "mlp"
    in_dim: int
    out_dim: int
    width: int = 128
    depth: int = 3
    epochs: int = 50
    batch: int = 64
    lr: float = 1e-2
    optimizer: str = "sgd"     # "sgd", "sgd_cosine", "adam"
    weight_decay: float = 0.0
    device: str = "cpu"
    # LOCCO
    locco: LoccoConfig = field(default_factory=LoccoConfig)
    # Data noise / extrema
    label_noise: float = 0.0   # fraction to flip labels (clf) or add extra noise (reg)
    extreme_lr: Optional[float] = None   # override lr
    extreme_batch: Optional[int] = None  # override batch

def build_model(cfg: RunConfig):
    if cfg.model_type == "linear":
        return LinearModel(cfg.in_dim, cfg.out_dim)
    elif cfg.model_type == "mlp":
        return MLP(cfg.in_dim, cfg.out_dim, width=cfg.width, depth=cfg.depth, bn=False)
    else:
        raise ValueError("unknown model_type")

def build_optimizer(cfg: RunConfig, model: nn.Module):
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.0)
        sched = None
    elif cfg.optimizer == "sgd_cosine":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = None
    else:
        raise ValueError("unknown optimizer")
    return opt, sched

def maybe_corrupt_labels(y: np.ndarray, frac: float, num_classes: int, seed=0):
    if frac <= 0 or y.ndim == 2: return y
    rng = np.random.default_rng(seed)
    y_new = y.copy()
    n = len(y_new); k = int(frac * n)
    idx = rng.choice(n, size=k, replace=False)
    y_new[idx] = rng.integers(0, num_classes, size=k)
    return y_new

def add_reg_noise(y: np.ndarray, std: float, seed=0):
    if std <= 0: return y
    rng = np.random.default_rng(seed)
    return y + rng.normal(scale=std, size=y.shape)

def run_one(cfg: RunConfig, train_loader, val_loader, test_loader):
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    opt, sched = build_optimizer(cfg, model)

    results = []
    # LOCCO init
    eta0 = cfg.extreme_lr if cfg.extreme_lr is not None else cfg.lr
    B0 = cfg.extreme_batch if cfg.extreme_batch is not None else cfg.batch
    locco = LoccoState(cfg.locco, eta0=eta0, B0=B0, loss0=1.0)
    eta, B = eta0, B0

    for ep in range(cfg.epochs):
        tr_loss, tr_acc, eta, B = train_one_epoch(
            model, train_loader, cfg.task, opt, device,
            locco_state=locco if cfg.locco.use else None,
            eta=eta, B=B, L_s=cfg.locco.Ls
        )
        if sched: sched.step()
        val_loss, val_acc = evaluate(model, val_loader, cfg.task, device)
        test_loss, test_acc = evaluate(model, test_loader, cfg.task, device)
        row = dict(epoch=ep, lr=eta, batch=B, train_loss=tr_loss,
                   val_loss=val_loss, test_loss=test_loss)
        if cfg.task == "clf":
            row.update(dict(train_acc=tr_acc, val_acc=val_acc, test_acc=test_acc))
        else:
            row.update(dict(train_rmse=math.sqrt(tr_loss), val_rmse=math.sqrt(val_loss),
                            test_rmse=math.sqrt(test_loss)))
        results.append(row)
        print(f"[{cfg.name}] ep {ep:03d} | lr={eta:.4g} B={B:4d} | tr={tr_loss:.4f} val={val_loss:.4f} test={test_loss:.4f}")

    return results, model

# -------------- Plots & Logging ----------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def write_csv(rows: List[Dict[str, Any]], path: str):
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    # Collect all unique keys from all rows
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    # Sort fieldnames for consistency
    fieldnames = sorted(fieldnames)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def plot_curves(rows: List[Dict[str, Any]], path: str, task: str, title: str):
    ensure_dir(os.path.dirname(path))
    xs = [r["epoch"] for r in rows]
    plt.figure(figsize=(7,4.2))
    plt.plot(xs, [r["train_loss"] for r in rows], label="train_loss")
    plt.plot(xs, [r["val_loss"] for r in rows], label="val_loss")
    plt.plot(xs, [r["test_loss"] for r in rows], label="test_loss")
    ax2 = plt.gca().twinx()
    ax2.plot(xs, [r["lr"] for r in rows], linestyle="--", alpha=0.6, label="lr")
    ax2.plot(xs, [r["batch"] for r in rows], linestyle=":", alpha=0.6, label="batch")
    plt.title(title); plt.xlabel("epoch"); plt.legend(loc="upper right")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines+lines2, labels+labels2, loc="upper right")
    plt.tight_layout(); plt.savefig(path); plt.close()

# -------------- Suite: POC, Benchmarks, Ablations, Extrema ----------------
def split_dataset(X, y, splits=(0.6, 0.2, 0.2), shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    if shuffle: rng.shuffle(idx)
    n1 = int(splits[0]*n); n2 = int(splits[1]*n)
    id_tr, id_val, id_te = idx[:n1], idx[n1:n1+n2], idx[n1+n2:]
    return (X[id_tr], y[id_tr]), (X[id_val], y[id_val]), (X[id_te], y[id_te])

def build_loaders_for_task(task: str, noise: float, batch: int, seed=0):
    if task == "clf_moons":
        X, y = make_moons(n=800, noise=0.25, seed=seed)
        y = maybe_corrupt_labels(y, frac=noise, num_classes=2, seed=seed+1)
        in_dim, out_dim = X.shape[1], 2
    elif task == "clf_blobs":
        X, y = make_blobs(n=3000, k=7, dim=12, std=1.8, seed=seed)
        y = maybe_corrupt_labels(y, frac=noise, num_classes=7, seed=seed+1)
        in_dim, out_dim = X.shape[1], 7
    elif task == "reg_sine":
        X, y = make_regression_sine(n=2000, noise=0.2+noise, seed=seed)
        in_dim, out_dim = X.shape[1], 1
    elif task == "reg_linear":
        X, y = make_regression_linear(n=2000, dim=25, noise=0.6+2*noise, seed=seed)
        in_dim, out_dim = X.shape[1], 1
    else:
        raise ValueError("unknown task")
    (Xtr,ytr),(Xv,yv),(Xt,yt) = split_dataset(X,y,seed=seed)
    train_loader = make_loader(Xtr,ytr,batch)
    val_loader   = make_loader(Xv,yv,batch,shuffle=False)
    test_loader  = make_loader(Xt,yt,batch,shuffle=False)
    # Map to unified task label
    task_kind = "clf" if "clf" in task else "reg"
    return (train_loader, val_loader, test_loader, in_dim, out_dim, task_kind)

def suite(args):
    set_seed(args.seed)
    results_dir = "results"; ensure_dir(results_dir)

    experiments: List[RunConfig] = []

    # ---------- POC: small classification + regression ----------
    for task_name in ["clf_moons", "reg_sine"]:
        train_loader, val_loader, test_loader, in_dim, out_dim, task_kind = \
            build_loaders_for_task(task_name, noise=0.0, batch=args.batch, seed=args.seed)
        # Baseline: linear / SGD
        experiments.append(RunConfig(
            name=f"POC-{task_name}-linear-sgd",
            task=task_kind, model_type="linear", in_dim=in_dim, out_dim=out_dim,
            epochs=30, batch=args.batch, lr=0.05 if task_kind=="clf" else 0.02,
            optimizer="sgd", locco=LoccoConfig(use=False)
        ))
        # Deep: MLP / Adam
        experiments.append(RunConfig(
            name=f"POC-{task_name}-mlp-adam",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=40, width=128, depth=3, batch=args.batch, lr=0.01,
            optimizer="adam", locco=LoccoConfig(use=False)
        ))
        # LOCCO (adaptive batch, optional LR guard off for clarity)
        experiments.append(RunConfig(
            name=f"POC-{task_name}-mlp-locco",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=40, width=128, depth=3, batch=args.batch, lr=0.01,
            optimizer="sgd", locco=LoccoConfig(
                use=True, alpha=0.05, beta=0.0, rho=0.9, Ls=1.0,
                use_curv_proxy=True, preserve_eta_over_B=False,
                B_min=8, B_max=1024
            )
        ))

    # ---------- Benchmarks: broader tasks ----------
    for task_name in ["clf_blobs", "reg_linear"]:
        train_loader, val_loader, test_loader, in_dim, out_dim, task_kind = \
            build_loaders_for_task(task_name, noise=0.0, batch=args.batch, seed=args.seed)
        # SGD cosine
        experiments.append(RunConfig(
            name=f"BM-{task_name}-mlp-sgdcos",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=60, width=256, depth=4, batch=args.batch, lr=0.05 if task_kind=="clf" else 0.02,
            optimizer="sgd_cosine", locco=LoccoConfig(use=False)
        ))
        # Adam baseline
        experiments.append(RunConfig(
            name=f"BM-{task_name}-mlp-adam",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=60, width=256, depth=4, batch=args.batch, lr=0.01,
            optimizer="adam", locco=LoccoConfig(use=False)
        ))
        # LOCCO strong
        experiments.append(RunConfig(
            name=f"BM-{task_name}-mlp-locco",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=60, width=256, depth=4, batch=args.batch, lr=0.01,
            optimizer="sgd", locco=LoccoConfig(
                use=True, alpha=0.03, beta=0.0, rho=0.9, Ls=1.0,
                use_curv_proxy=True, preserve_eta_over_B=True,
                B_min=16, B_max=2048
            )
        ))

    # ---------- Ablations ----------
    # LOCCO w/ no smoothing; no curvature proxy; fixed batch only
    for ab in ["nosmooth", "nocurv", "fixedB"]:
        task_name = "clf_blobs"
        train_loader, val_loader, test_loader, in_dim, out_dim, task_kind = \
            build_loaders_for_task(task_name, noise=0.0, batch=args.batch, seed=args.seed)
        if ab == "nosmooth":
            locco = LoccoConfig(use=True, alpha=0.05, beta=0.0, rho=0.0, Ls=1.0,
                                use_curv_proxy=True, preserve_eta_over_B=False,
                                B_min=8, B_max=1024)
            name = "ABL-clf_blobs-locco-nosmooth"
        elif ab == "nocurv":
            locco = LoccoConfig(use=True, alpha=0.05, beta=0.0, rho=0.9, Ls=1.0,
                                use_curv_proxy=False, preserve_eta_over_B=False,
                                B_min=8, B_max=1024)
            name = "ABL-clf_blobs-locco-nocurv"
        else:  # fixedB
            locco = LoccoConfig(use=False)
            name = "ABL-clf_blobs-fixedB"
        experiments.append(RunConfig(
            name=name, task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=50, width=256, depth=3, batch=args.batch, lr=0.02,
            optimizer="sgd", locco=locco
        ))

    # ---------- Extrema tests ----------
    for (lr, B, lab_noise) in [(1e-4, 8, 0.0), (1.0, 8, 0.0), (0.02, 2048, 0.0), (0.02, 64, 0.3)]:
        task_name = "clf_moons"
        train_loader, val_loader, test_loader, in_dim, out_dim, task_kind = \
            build_loaders_for_task(task_name, noise=lab_noise, batch=B, seed=args.seed)
        experiments.append(RunConfig(
            name=f"EXT-{task_name}-mlp-lr{lr}-B{B}-noise{lab_noise}",
            task=task_kind, model_type="mlp", in_dim=in_dim, out_dim=out_dim,
            epochs=40, width=128, depth=3, batch=B, lr=lr,
            optimizer="sgd", locco=LoccoConfig(use=True, alpha=0.05, beta=0.0,
                                               rho=0.9, Ls=1.0, use_curv_proxy=True,
                                               preserve_eta_over_B=False, B_min=4, B_max=4096),
            extreme_lr=lr, extreme_batch=B, label_noise=lab_noise
        ))

    # ---------- Run selected or all ----------
    if not args.all:
        experiments = experiments[:5]  # quick POC subset

    summary_rows = []
    for cfg in experiments:
        print("\n==== Running:", cfg.name)
        # Rebuild loaders per-experiment (some params differ)
        task_name = "clf_moons" if cfg.in_dim == 2 and cfg.out_dim == 2 and cfg.task=="clf" else None
        # Reload from builder to ensure repeats are fair
        # (For simplicity, reuse earlier design choices:)
        # Use batch from cfg (may be overridden by LOCCO later)
        # Note: noise already incorporated in builder calls above; here we regenerate with 0 noise for fairness
        # But to keep deterministic, we build again with zero noise unless the name contains "EXT-"
        if cfg.name.startswith("POC-"):
            task_internal = "clf_moons" if cfg.task=="clf" and cfg.in_dim==2 else ("reg_sine" if cfg.task=="reg" else None)
        elif cfg.name.startswith("BM-") and cfg.task=="clf":
            task_internal = "clf_blobs"
        elif cfg.name.startswith("BM-") and cfg.task=="reg":
            task_internal = "reg_linear"
        elif cfg.name.startswith("ABL-"):
            task_internal = "clf_blobs"
        else:
            task_internal = "clf_moons" if cfg.task=="clf" else "reg_sine"

        noise = 0.0
        if cfg.name.startswith("EXT-"):
            # infer from name; already carried in cfg
            noise = cfg.label_noise

        if task_internal is None:
            raise ValueError(f"Could not determine task_internal for config: {cfg.name}")
        train_loader, val_loader, test_loader, in_dim2, out_dim2, task_kind2 = \
            build_loaders_for_task(task_internal, noise=noise, batch=cfg.batch, seed=args.seed)

        # Run
        rows, model = run_one(cfg, train_loader, val_loader, test_loader)

        # Persist
        csv_path = os.path.join(results_dir, f"{cfg.name}.csv")
        plot_path = os.path.join(results_dir, f"{cfg.name}.png")
        write_csv(rows, csv_path)
        title = f"{cfg.name} ({cfg.optimizer})"
        plot_curves(rows, plot_path, cfg.task, title)

        # Summaries
        best_val = min(rows, key=lambda r: r["val_loss"])
        final = rows[-1]
        out = dict(
            name=cfg.name, task=cfg.task, model=cfg.model_type,
            best_val_loss=best_val["val_loss"], final_test_loss=final["test_loss"],
            best_val_epoch=best_val["epoch"], final_lr=final["lr"], final_batch=final["batch"]
        )
        if cfg.task=="clf":
            out = out | dict(best_val_acc=best_val["val_acc"], final_test_acc=final["test_acc"])
        else:
            out = out | dict(best_val_rmse=math.sqrt(best_val["val_loss"]),
                            final_test_rmse=math.sqrt(final["test_loss"]))
        summary_rows.append(out)

    # Write experiment summary
    write_csv(summary_rows, os.path.join(results_dir, "SUMMARY.csv"))
    print("\nWrote per-run logs and plots to ./results/. Summary in SUMMARY.csv")

# -------------- CLI ----------------

# Remove CLI args, use RUN_TYPE and RUN_SEED instead
class Args:
    def __init__(self):
        self.seed = RUN_SEED
        self.batch = 64
        # If you want to run the full suite, set RUN_TYPE = "ALL"
        self.all = (RUN_TYPE.upper() == "ALL")

if __name__ == "__main__":
    args = Args()
    suite(args)
