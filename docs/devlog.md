# Devlog

Engineering journal for the rewrite: decisions, numbers, and **what broke**.
Each entry: *what / why / failures / talking points*. Tracked in [#1](https://github.com/kaps-c/Vision-World-Models-Snake/issues/1).

---

## Context

Clean-room rewrite of my L=14 transformer-world-model / strict-JEPA track from the
Snekboros team project (SUTD, Apr 2026). The original code was lost when the team
repo was cleaned after the deadline. The spec is my own extended-grid write-up:
three pipelines (DINOv2-WM, JEPA + projector + VICReg, identity), linear head
probes, a rank-ratio kill-switch, a TPE search, and rollout GIFs.

**Interview framing:** "I rebuilt it from my own spec, and used the rewrite to make it
properly tested and reproducible. Original numbers are reference targets, not claims."

---

## Slice 0 — Scaffold (2026-10-12)

**What.** `src/` layout package `snekwm`, `pyproject.toml`, ruff (PEP 8 + pydocstyle
Google convention), pytest, GitHub Actions CI with CPU torch.

**Failures.**
| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| `git add` refused `src/snekwm/data` | `.gitignore` had `data/` meant for the dataset dir; unanchored patterns match at any depth | Anchor: `/data/`, `/results/` | Anchor root-only ignore patterns |
| Scaffold landed inside the `feat(env)` commit | The failed `git add` had already staged the other paths; the `&&` chain skipped only the first commit | Rebuilt the 2 unpushed commits | Check `git status` after a failed add; a non-zero exit doesn't mean "nothing staged" |
| ruff D413 on every `Args:` block | `select = ["D"]` without a convention enables contradictory rules | `[tool.ruff.lint.pydocstyle] convention = "google"` | Pick a docstring convention up front |
| Stale `snekboros` package in the conda env | Editable install pointing at the deleted original clone | Uninstalled; installed `snekwm` editable | — |

---

## Slice 1 — Simulator

**What.** `SnakeEnv`: L×L grid, 5 absolute actions `{(0,0),(0,1),(0,-1),(-1,0),(1,0)}`,
immutable `State`, RGB render at `cell_px=16` → 224×224 for L=14 (DINOv2's native
resolution, 16×16 = 256 ViT-S/14 patches).

**Decisions.**
- `(0,0)` = keep heading; a reverse into the neck is **ignored** (a common Snake rule,
  and it avoids a trivial "suicide" action).
- Frozen-dataclass `State` makes the sim easy to test (construct any position directly)
  and safe to branch from during planning.
- Moving into the cell the tail is vacating is legal (tested explicitly; a classic
  off-by-one in Snake implementations).

**Talking point.** 14 behaviour tests (walls, self-collision, tail-chase, win, render
colours, seeded determinism) written before the implementation.

---

## Slice 2 — Data: Hamiltonian policy, collection, dataset

**What.**
- `hamiltonian_cycle(L)`: zig-zag cycle; raises for odd L (no Hamiltonian cycle exists
  on an odd grid, which is a colouring/parity argument).
- `PerturbedHamiltonian(ε)`: follows the cycle and takes a random **safe** action with
  probability ε. If the cycle move is fatal (after being knocked off the cycle), it
  takes a safe move instead.
- `snek collect`: episodes to `.npz`. `TransitionDataset` yields `(obs_t, a_t, obs_t+1)`
  plus head labels.

**Decisions.**
- **Store frames at grid resolution, upsample on load.** Rendering is pure pixel
  repetition, so this is lossless. 32 episodes × 300 steps = **132 KB** vs about 1.4 GB
  at 224 px.
- **Split by episode, not by transition.** Consecutive frames are near-duplicates, so a
  random transition split leaks and inflates val/probe scores.
- Movement rules live in pure functions (`next_head`, `would_die`) shared by the sim
  and the policy, so the policy's notion of "safe" can't drift from the simulator's.
- The direction-of-travel check (`neck == succ(head)` → walk the cycle backwards) makes
  ε=0 provably safe from any length-2 spawn. Tested: 20 seeds on 4×4 all **win**.

**Dataset stats** (L=14, 32 eps, ε=0.1, cap 300): 9,600 transitions, 0 deaths.
Actions: KEEP 2%, RIGHT 8%, LEFT 8%, **UP 40%, DOWN 42%**.

**Watch-outs (predicted failure modes, to check in later slices).**
- **Action imbalance.** The zig-zag cycle mostly moves vertically. A predictor can
  reach low loss while largely ignoring the action token. This is relevant to the
  "no-action ablation collapses" finding: check if it's partly a data artefact.
- **Action aliasing.** KEEP and "the current heading" have identical effects, so the
  action→outcome map is not one-to-one.
- **No death transitions** at ε=0.1. That's fine for head probes, but the WM never sees
  a terminal frame.
- ε isn't in the original write-up; 0.1 is my default.

---

## Slice 2b — Fix: the data policy didn't match the spec

**Failure.** Slice 2's policy followed one fixed zig-zag cycle plus ε-noise.
Re-reading the spec: my original track used "the same PerturbedHamiltonianCycle
generator as the main system", which (in the team code) means a **random cycle per
episode** plus **safe shortcuts toward food**, not ε-noise alone. The tell was the
action histogram: **82% UP/DOWN**, because a fixed zig-zag cycle is mostly vertical.

**Fix (own implementation of the idea).**
- `random_hamiltonian_cycle`: tile the grid with 2×2 loops, draw a random spanning
  tree over the blocks (randomised Prim), and bridge adjacent loops along tree edges.
  Merging loops along a *tree* can't create a second cycle, so the result is
  provably one Hamiltonian cycle. Tested on 2/4/14 grids × 5 seeds.
- Safe shortcuts: jump to a neighbour further ahead on the cycle only if it is
  (a) before the tail (`0 < d < tail_gap`), (b) not past the food, and (c) the body
  is still in cycle order. That keeps the never-trap guarantee: ε=0 wins on 4×4 and
  6×6 for 20 seeds each.
- Kept ε (the write-up's GIF oracle uses "ε=0", so my version had one). The data ε
  is unknown, so the default is 0.1.

**Before → after** (L=14, 32 eps, ε=0.1):
| | fixed cycle | random backbone + shortcuts |
|---|---|---|
| UP/DOWN share | 82% | 50% (≈25% each direction) |
| deaths | 0/32 | 7/32 |
| final length | — | ≈12 (spec: ~14–16 by t=300) |

**Why deaths appeared.** ε-noise can break cycle order. Shortcuts are then disabled
and the policy falls back to "follow the cycle, or any safe move", and a safe-now
move can still walk into a dead end. That's acceptable: the WM now sees terminal frames.

**Test that caught a weak test.** My first shortcut test asserted "first food in
< 64 steps" on 8×8. Pure cycle-following *also* always satisfies that (cycle length
is 64), so the test couldn't fail. I replaced it with a mean bound (< 16 vs ≈32 for
pure cycle). **Lesson:** ask "would this test fail without the feature?"

**Talking point.** "I found the data was 82% vertical moves, which would let a world
model ignore actions and still score well. It led me back to a spec mismatch."

---

## Slice 3 — Frozen DINOv2, identity floor, linear probes

**What.**
- `FrozenDinov2`: `torch.hub` ViT-S/14, ImageNet normalisation, frozen, and stays in eval
  mode even under `.train()`. The backbone is injectable, so unit tests use a fake one
  (no download in CI). The real-weights test is marked `network`.
- `encode_episodes`: caches `(T+1, 256, 384)` fp16 tokens per episode (1.7 GB for
  9.2k frames; about 85 s on an RTX 3070 Ti Laptop).
- `fit_probe`: StandardScaler + RidgeCV (α ∈ logspace(−3, 4)) on **mean-pooled**
  tokens, R² per coordinate, and `r2_joint` = uniform mean. Fit on train episodes,
  scored on held-out episodes.
- `snek probe` writes `results/dinov2_frozen/{config.json, probe.json, train.log}`.

**Result.**
| Probe | R²_x | R²_y | R²_joint | Original write-up |
|---|---|---|---|---|
| Frozen DINOv2 encoder (`z_t → head_t`) | 0.946 | 0.917 | **0.932** | 0.927 ✓ |
| Identity predictor (`ẑ_{t+1}=z_t → head_{t+1}`) | 0.925 | 0.890 | **0.907** | not reported |

**Finding: the predictor probe, as specified, is nearly saturated by a no-op.**
- Predicting `head_{t+1} = head_t` *exactly*, with no model at all, scores
  **R² = 0.968**, because the head moves at most one cell out of 14 per step.
- So the identity WM gets 0.907 on the "does the predictor know where the snake will be"
  probe while knowing **nothing** about dynamics.
- Consequence for the original numbers: predictor probes of 0.80 (frozen-DINOv2 JEPA)
  and 0.32 (SmallViT recipe) sit **below** the identity floor in this setup. A low
  predictor-probe score therefore indicates lost *position* information, not
  necessarily missing *dynamics*. The original "no-action → 0.04" collapse is still a
  real failure, but the metric can't show what a *working* predictor learned.
- **Proposed fix:** also probe the **displacement** `Δhead = head_{t+1} − head_t`. Predicting
  zero displacement scores R² ≈ 0.000, so the floor sits at zero and any positive
  score is dynamics knowledge.

**Talking point.** "My reproduced ceiling matched the paper to 0.005. The identity
baseline showed the paper's predictor metric was dominated by a copy-the-input
solution, so I added a displacement probe whose trivial baseline is zero."

**Small failures.**
- ruff's pep8-naming flagged sklearn's `X` convention. I whitelisted `X`/`X_*`
  instead of renaming, because matching the ecosystem idiom beats the linter here.
- Windows keeps `train.log` locked while the logger holds it, so `RunDir.close()`
  releases the handlers (otherwise test tmp dirs can't be cleaned).

### Slice 3b — Displacement probe (and a wrong prediction of mine)

Added `displacement_probe_data` (`ẑ_{t+1} → head_{t+1} − head_t`) and reported it for
the identity WM.

**I predicted the identity floor would be ≈ 0. It is 0.391.** The ≈ 0 figure was for
*predicting zero displacement*. A *probe* on `z_t` can do better: the frame shows the
heading (neck → head), and the data policy is predictable (it mostly continues along
its cycle). So ~40% of displacement variance is guessable from the current state with
no action.

| Probe (identity WM) | R²_joint | Headroom to a perfect model |
|---|---|---|
| predictor (`→ head_{t+1}`) | 0.907 | 0.09 |
| displacement (`→ Δhead`) | **0.391** | **0.61** |

**Reading.** Displacement is fully determined by (heading, action), so a WM that uses the
action can approach 1.0. Any score above 0.391 is attributable to the action/dynamics.
The metric now has about 7× more dynamic range than the original one.

**Lesson.** State the baseline's exact definition before predicting its value: "predict
zero" and "best linear guess from the current state" are different floors.

---

## Slice 4 — DINOv2-WM (predictor on frozen DINOv2 features)

**What.**
- `ActionConditionedPredictor`: a pre-norm transformer over 256 patch tokens plus one
  learned **action token**, with depth 4, dim 384 (7.5M params). The window is Markovian
  (one frame), so no block-causal mask is needed; that's documented in the class.
  Flags: `residual` (predict `z_t + f`, with `f` zero-initialised so it *starts as* the
  identity WM; this is tested) and `use_action` (for the slice 5 ablation).
- `FeatureTransitions`: the whole cached split lives **on the GPU** (≈1.4 GB fp16), and
  batches are indexed straight from it, with no DataLoader overhead.
- Training: MSE in DINOv2 space, AdamW 3e-4, wd 0.05, OneCycle, bf16 autocast, grad clip
  1.0, 5 epochs (the write-up's matched budget). Per epoch it logs val MSE/cosine
  *next to the identity WM's*.
- `snek wm --config configs/dinov2_wm.toml` → `results/dinov2_wm/{config,history,probe}.json,
  final.pt, train.log`.

**Results** (val split: 6 held-out episodes, 1,800 transitions).
| Model | Val MSE ↓ | Val cos ↑ | Predictor probe | Displacement probe |
|---|---|---|---|---|
| Identity WM | 0.223 (write-up: 0.221 ✓) | 0.984 | 0.907 | 0.391 |
| DINOv2-WM (direct) | 0.243 | 0.983 | 0.923 | **0.838** |
| DINOv2-WM (residual) | **0.134** | **0.990** | 0.929 | 0.556 |

### Finding 1: "loses to identity on MSE" ≠ "learned nothing"
The direct model never beats copy-the-input on MSE (0.243 vs 0.223). This reproduces the
write-up, where identity also won raw MSE. **But its displacement probe jumps from
0.39 to 0.84**, so it clearly learned action-conditioned dynamics. Why the MSE gap: one
cell changes per step, so roughly 250/256 patches are near-static. A direct predictor
must *re-synthesise* all of them and pays a small error on each, while the identity WM
gets them free.

### Finding 2: residual parameterisation beats identity, but the probe says it's worse
I hypothesised that starting at identity (residual) would beat identity on MSE while keeping
the dynamics. MSE improved by 40%, **but the displacement probe dropped to 0.556**. Two
explanations: (a) a probe artefact, where the change is small relative to `z_t` and drowns
under mean-pooling; or (b) the residual model really learns less dynamics.

**Discriminating test:** probe the predicted change `ẑ_{t+1} − z_t` directly.
| | ‖Δ‖ / ‖z‖ | Probe on Δ |
|---|---|---|
| direct | 0.149 | 0.771 |
| residual | 0.108 | **0.871** |

→ **(a).** The residual model encodes the dynamics *better*; mean-pooling the full
prediction hid it. The identity WM's Δ is exactly 0, so this "Δ-probe" has a true zero
floor, which is the property I wanted in slice 3b.

**Lesson.** A probe measures the representation *and* the readout. When two metrics
disagree, design one test that separates the hypotheses before believing either.

**Talking point.** "Three metrics ranked the same two models three different ways. MSE
favoured residual, the pooled probe favoured direct, and the Δ-probe favoured residual. I
traced it to mean-pooling scale and picked the metric whose floor is genuinely zero."

**Observations (unexplained, not chased).** Some epochs took about 4 min instead of
about 35 s (epoch 3 of run 1, epoch 1 of run 2). The laptop GPU was idling at 330 MHz
afterwards, so it's likely a power-state or thermal issue or another process. The
results are unaffected (seeded), but it matters for any wall-clock comparison.
