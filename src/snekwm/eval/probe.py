"""Linear probes: how linearly decodable is the snake's head from features?

Probe families (all on mean-pooled patch tokens, fit on train episodes and
scored on held-out episodes):

* **encoder**: ``z_t -> head_t``. Does the encoder know where the snake is?
* **predictor**: ``z_hat_{t+1} -> head_{t+1}``. Saturated: copying ``head_t``
  already scores R^2 ~ 0.97 because the head moves one cell per step.
* **displacement** / **change**: ``z_hat_{t+1}`` (or ``z_hat_{t+1} - z_t``) ->
  ``head_{t+1} - head_t``. *Caveat:* actions are absolute directions, so a
  predictor that merely passes its action token through scores as well as
  :func:`action_only_control` (~0.88). Kept for comparability only.
* **transfer** (the dynamics metric): fit the head probe on *real*
  encodings only, freeze it, and decode the *predicted* latents
  (:func:`transfer_probe`). Action passthrough cannot help; the prediction
  has to land where the true next frame's encoding lies.

``r2_joint`` is the uniform average of the per-coordinate R^2.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline
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
    model = _fit(X_train, y_train)
    return _score(y_val, model.predict(X_val), model, len(X_train))


def _fit(X: np.ndarray, y: np.ndarray) -> Pipeline:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)


def _score(
    y: np.ndarray, pred: np.ndarray, model: Pipeline, n_train: int
) -> ProbeResult:
    r2 = r2_score(y, pred, multioutput="raw_values")
    return ProbeResult(
        r2_x=float(r2[0]),
        r2_y=float(r2[1]),
        r2_joint=float(r2.mean()),
        alpha=float(model[-1].alpha_),
        n_train=n_train,
        n_val=len(y),
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


def change_tensors(
    preds: Mapping[str, torch.Tensor], feats: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return predicted changes ``z_hat_{t+1} - z_t`` per episode, ``(T, N, D)``."""
    return {k: preds[k].float() - feats[k][:-1].float() for k in preds}


def transfer_probe(
    preds: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    episodes: Mapping[str, dict[str, np.ndarray]],
    train: Sequence[str],
    val: Sequence[str],
    normalize: bool = False,
) -> dict[str, ProbeResult]:
    """Decode predicted latents with a head probe fit on real encodings only.

    Args:
        preds: ``{name: (T, N, D)}`` predictions of the next latent.
        reference: ``{name: (T+1, N, D)}`` real encodings in the same space.
        episodes: Episode arrays keyed by name.
        train: Episodes used to fit the probe (on ``reference`` only).
        val: Held-out episodes the predictions are scored on.
        normalize: L2-normalise every token of both ``preds`` and ``reference``
            first. Use it when the model was trained with a scale-invariant loss
            (JEPA's normalised invariance), which never constrains the scale of
            its predictions; decoding raw predictions would then penalise a
            quantity the model was never asked to match.

    Returns:
        ``transfer``: decoded ``P(z_hat_{t+1})`` vs ``head_{t+1}`` (identity
        world model ~ copy-the-head floor; perfect model ~ encoder probe).
        ``transfer_displacement``: ``P(z_hat_{t+1}) - P(z_t)`` vs the true
        displacement (identity world model scores 0).
    """
    if normalize:
        preds = {
            k: torch.nn.functional.normalize(v.float(), dim=-1)
            for k, v in preds.items()
        }
        reference = {
            k: torch.nn.functional.normalize(v.float(), dim=-1)
            for k, v in reference.items()
        }
    X, y = zip(
        *(encoder_probe_data(reference[n], episodes[n]) for n in train), strict=True
    )
    model = _fit(np.concatenate(X), np.concatenate(y))
    nxt, now, true_next, true_now = [], [], [], []
    for n in val:
        ep, mask = episodes[n], episodes[n]["alive"][1:]
        nxt.append(model.predict(_pool(preds[n])[mask]))
        now.append(model.predict(_pool(reference[n][:-1])[mask]))
        true_next.append(ep["heads"][1:][mask])
        true_now.append(ep["heads"][:-1][mask])
    nxt, now = np.concatenate(nxt), np.concatenate(now)
    true_next = np.concatenate(true_next).astype(np.float64)
    true_now = np.concatenate(true_now).astype(np.float64)
    n_train = sum(len(x) for x in X)
    return {
        "transfer": _score(true_next, nxt, model, n_train),
        "transfer_displacement": _score(
            true_next - true_now, nxt - now, model, n_train
        ),
    }


def action_only_control(
    episodes: Mapping[str, dict[str, np.ndarray]],
    train: Sequence[str],
    val: Sequence[str],
) -> ProbeResult:
    """Displacement R^2 from a one-hot of the action alone (no image)."""

    def xy(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for n in names:
            ep, mask = episodes[n], episodes[n]["alive"][1:]
            xs.append(np.eye(5)[ep["actions"]][mask])
            ys.append((ep["heads"][1:] - ep["heads"][:-1])[mask])
        return np.concatenate(xs), np.concatenate(ys).astype(np.float64)

    return fit_probe(*xy(train), *xy(val))


def _pool(tokens: torch.Tensor) -> np.ndarray:
    return tokens.float().mean(dim=1).numpy()
