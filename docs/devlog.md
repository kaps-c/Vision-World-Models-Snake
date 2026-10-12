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
