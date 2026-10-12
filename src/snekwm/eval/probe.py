"""Linear probes: how linearly decodable is the snake's head from features?

Two probe families (both fit on mean-pooled patch tokens):

* **encoder probe**: ``z_t -> head_t`` (does the encoder know where the snake is?)
* **predictor probe**: ``z_hat_{t+1} -> head_{t+1}`` (does the world model know
  where the snake *will* be?)
* **displacement probe**: ``z_hat_{t+1} -> head_{t+1} - head_t``. The head moves
  one cell per step, so copying ``head_t`` already scores R^2 ~ 0.97 on the
  predictor probe; on displacement that no-op scores ~0, so a positive score
  isolates knowledge of the dynamics.

``r2_joint`` is the uniform average of the per-coordinate R^2.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 4, 15)


@dataclass(frozen=True)
class ProbeResult:
    """Held-out R^2 of a ridge probe."""

    r2_x: float
    r2_y: float
    r2_joint: float
    alpha: float
    n_train: int
    n_val: int

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly dict."""
        return asdict(self)


def fit_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> ProbeResult:
    """Fit standardised RidgeCV on train, report R^2 on val."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    r2 = r2_score(y_val, pred, multioutput="raw_values")
    return ProbeResult(
        r2_x=float(r2[0]),
        r2_y=float(r2[1]),
        r2_joint=float(r2.mean()),
        alpha=float(model[-1].alpha_),
        n_train=len(X_train),
        n_val=len(X_val),
    )


PairFn = Callable[[torch.Tensor, dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]]


def evaluate(
    pair_fn: PairFn,
    tensors: Mapping[str, torch.Tensor],
    episodes: Mapping[str, dict[str, np.ndarray]],
    train: Sequence[str],
    val: Sequence[str],
) -> ProbeResult:
    """Build probe data with ``pair_fn`` per episode, fit on ``train``, score ``val``.

    Episodes are keyed by name; splitting by episode avoids leakage between
    near-duplicate consecutive frames.
    """

    def stack(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        pairs = [pair_fn(tensors[n], episodes[n]) for n in names]
        return np.concatenate([p[0] for p in pairs]), np.concatenate(
            [p[1] for p in pairs]
        )

    return fit_probe(*stack(train), *stack(val))


def encoder_probe_data(
    feats: torch.Tensor, episode: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair mean-pooled ``(T+1, N, D)`` frame features with head positions."""
    mask = episode["alive"]
    return _pool(feats)[mask], episode["heads"][mask].astype(np.float64)


def predictor_probe_data(
    preds: torch.Tensor, episode: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ``(T, N, D)`` next-step predictions with next head positions."""
    mask = episode["alive"][1:]
    return _pool(preds)[mask], episode["heads"][1:][mask].astype(np.float64)


def displacement_probe_data(
    preds: torch.Tensor, episode: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ``(T, N, D)`` next-step predictions with head displacements."""
    mask = episode["alive"][1:]
    delta = episode["heads"][1:] - episode["heads"][:-1]
    return _pool(preds)[mask], delta[mask].astype(np.float64)


def _pool(tokens: torch.Tensor) -> np.ndarray:
    return tokens.float().mean(dim=1).numpy()
