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
