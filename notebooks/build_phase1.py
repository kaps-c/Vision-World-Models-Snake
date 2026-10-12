"""Build ``phase1_frozen_and_dinov2_wm_analysis.ipynb`` (run, then execute it).

Keeping the notebook's source in a script makes diffs reviewable; the executed
``.ipynb`` (with outputs) is what gets committed for reading on GitHub.
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# Phase 1 — Frozen DINOv2 and the DINOv2 world model (L=14)\n\n"
        "**Questions.** (1) How linearly decodable is the snake's head from frozen "
        "DINOv2 features (the *ceiling*)? (2) Does a world model trained on those "
        "features learn the action-conditioned dynamics, and how do we tell?\n\n"
        "**Setup.** 32 episodes × ≤300 steps from the random-backbone Hamiltonian "
        "policy (ε = 0.1) on a 14×14 grid, rendered at 224 px. Train/val split "
        "by episode (26/6). All numbers below are on the 6 held-out episodes.\n\n"
        "Re-run everything with:\n"
        "```\nsnek collect --config configs/collect.toml\n"
        "snek probe --config configs/probe_dinov2.toml\n"
        "snek wm --config configs/dinov2_wm.toml\n"
        "snek wm --config configs/dinov2_wm_residual.toml\n```"
    ),
    code(
        "import json\n"
        "from collections import Counter\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from snekwm.data.collect import load_episode\n"
        "from snekwm.env.simulator import ACTIONS\n"
        "from snekwm.eval import style\n\n"
        "style.apply()\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "RESULTS, FIGS = ROOT / 'results', ROOT / 'notebooks' / 'figures'\n"
        "FIGS.mkdir(parents=True, exist_ok=True)\n\n"
        "def load(run, name):\n"
        '    """Load ``results/<run>/<name>`` as JSON."""\n'
        "    return json.loads((RESULTS / run / name).read_text(encoding='utf-8'))"
    ),
    md("## 1. The data\n\nWhat the world model is trained on — and its biases."),
    code(
        "episodes = [load_episode(p) for p in sorted((ROOT / 'data/episodes').glob('*.npz'))]\n"
        "names = ['KEEP', 'RIGHT', 'LEFT', 'UP', 'DOWN']\n"
        "counts = Counter(a for e in episodes for a in e['actions'].tolist())\n"
        "total = sum(counts.values())\n"
        "stats = pd.DataFrame({\n"
        "    'action': names,\n"
        "    'count': [counts[i] for i in range(len(ACTIONS))],\n"
        "    'share': [f'{counts[i] / total:.1%}' for i in range(len(ACTIONS))],\n"
        "})\n"
        "lengths = [len(e['actions']) for e in episodes]\n"
        "deaths = sum(not e['alive'][-1] for e in episodes)\n"
        "print(f'{len(episodes)} episodes, {total} transitions, '\n"
        "      f'length {min(lengths)}–{max(lengths)} (mean {np.mean(lengths):.0f}), '\n"
        "      f'{deaths} deaths')\n"
        "stats"
    ),
    code(
        "ep = episodes[0]\n"
        "steps = np.linspace(0, len(ep['frames']) - 1, 6).astype(int)\n"
        "fig, axes = plt.subplots(1, 6, figsize=(10, 2))\n"
        "for ax, t in zip(axes, steps, strict=True):\n"
        "    ax.imshow(ep['frames'][t].repeat(16, 0).repeat(16, 1))\n"
        "    ax.set_title(f't = {t}', fontsize=8)\n"
        "    ax.axis('off')\n"
        "fig.suptitle('Episode 0 as the encoder sees it (224 × 224)', x=0.01, ha='left')\n"
        "fig.savefig(FIGS / 'phase1_filmstrip.png')\n"
        "plt.show()"
    ),
    md(
        "## 2. Probe results\n\n"
        "Ridge probes on mean-pooled patch tokens (held-out R²_joint, mean of x/y):\n\n"
        "| probe | features → target | trivial floor |\n|---|---|---|\n"
        "| encoder | `z_t → head_t` | — |\n"
        "| predictor | `ẑ_{t+1} → head_{t+1}` | copy `head_t` scores 0.97 |\n"
        "| displacement | `ẑ_{t+1} → Δhead` | state-only guess (identity WM) |\n"
        "| change | `ẑ_{t+1} − z_t → Δhead` | exactly 0 for the identity WM |"
    ),
    code(
        "frozen = load('dinov2_frozen', 'probe.json')\n"
        "runs = {'Direct WM': 'dinov2_wm', 'Residual WM': 'dinov2_wm_residual'}\n"
        "rows = {'Identity WM': {\n"
        "    'predictor': frozen['identity_predictor']['r2_joint'],\n"
        "    'displacement': frozen['identity_displacement']['r2_joint'],\n"
        "    'change': frozen['identity_change']['r2_joint'],\n"
        "}}\n"
        "for label, run in runs.items():\n"
        "    rows[label] = {k: v['r2_joint'] for k, v in load(run, 'probe.json').items()}\n"
        "probes = pd.DataFrame(rows).T[['predictor', 'displacement', 'change']]\n"
        "print(f\"Frozen DINOv2 encoder probe (ceiling): {frozen['encoder']['r2_joint']:.3f}\")\n"
        "probes.round(3)"
    ),
    code(
        "metrics = ['predictor', 'displacement', 'change']\n"
        "models = list(probes.index)\n"
        "x = np.arange(len(metrics))\n"
        "width = 0.26\n"
        "fig, ax = plt.subplots(figsize=(8, 3.6))\n"
        "for i, model in enumerate(models):\n"
        "    vals = probes.loc[model, metrics].to_numpy()\n"
        "    bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=style.SERIES[i],\n"
        "                  label=model)\n"
        "    for b, v in zip(bars, vals, strict=True):\n"
        "        ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2, max(v, 0)),\n"
        "                    textcoords='offset points', xytext=(0, 2), ha='center',\n"
        "                    fontsize=7, color=style.TEXT)\n"
        "ax.axhline(frozen['encoder']['r2_joint'], color=style.BASELINE, ls='--', lw=1.2)\n"
        "ax.annotate('encoder ceiling', (x[-1] + 0.45, frozen['encoder']['r2_joint']),\n"
        "            textcoords='offset points', xytext=(0, 3), ha='right', fontsize=8,\n"
        "            color=style.TEXT_MUTED)\n"
        "ax.set_xticks(x, metrics)\n"
        "ax.set_ylim(0, 1.08)\n"
        "ax.set_ylabel('held-out R²_joint')\n"
        "ax.set_title('Same three models, ranked differently by each probe', loc='left')\n"
        "ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))\n"
        "fig.savefig(FIGS / 'phase1_probes.png')\n"
        "plt.show()"
    ),
    md("## 3. Feature-space error: the world models against copying the input"),
    code(
        "hist = {label: pd.DataFrame(load(run, 'history.json')) for label, run in runs.items()}\n"
        "last = {label: h.iloc[-1] for label, h in hist.items()}\n"
        "errors = pd.DataFrame({\n"
        "    'Identity WM': {'val MSE': last['Direct WM']['identity_mse'],\n"
        "                    'val cos': last['Direct WM']['identity_cos']},\n"
        "    **{k: {'val MSE': v['val_mse'], 'val cos': v['val_cos']} for k, v in last.items()},\n"
        "}).T\n"
        "errors.round(4)"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6.5, 3.4))\n"
        "for i, (label, h) in enumerate(hist.items(), start=1):\n"
        "    ax.plot(h['epoch'], h['val_mse'], marker='o', color=style.SERIES[i], label=label)\n"
        "    ax.annotate(label, (h['epoch'].iloc[-1], h['val_mse'].iloc[-1]),\n"
        "                textcoords='offset points', xytext=(6, 0), va='center',\n"
        "                fontsize=8, color=style.TEXT)\n"
        "ident = hist['Direct WM']['identity_mse'].iloc[0]\n"
        "ax.axhline(ident, color=style.BASELINE, ls='--', lw=1.2)\n"
        "ax.annotate(f'identity WM {ident:.3f}', (1, ident), textcoords='offset points',\n"
        "            xytext=(0, 4), fontsize=8, color=style.TEXT_MUTED)\n"
        "ax.set(xlabel='epoch', ylabel='validation MSE (DINOv2 space)',\n"
        "       title='Only the residual model beats copying the input')\n"
        "style.log_axis(ax)\n"
        "ax.set_xticks(hist['Direct WM']['epoch'])\n"
        "ax.legend(loc='upper right')\n"
        "fig.savefig(FIGS / 'phase1_val_mse.png')\n"
        "plt.show()"
    ),
    md(
        "## 4. Findings\n\n"
        "1. **The ceiling reproduces.** Frozen DINOv2's encoder probe matches the "
        "original write-up (0.927) to within about 0.005.\n"
        "2. **The original predictor probe is saturated.** The identity WM, which knows "
        "nothing about dynamics, already scores about 0.91, because the head moves at "
        "most one cell per step. It can't rank world models.\n"
        "3. **MSE and dynamics disagree.** The direct WM never beats copy-the-input on "
        "MSE (as in the original), yet its displacement probe roughly doubles over "
        "identity. It learned the dynamics; it pays MSE re-synthesising ~250 static "
        "patches.\n"
        "4. **The pooled probe hides the residual model's dynamics.** Its displacement "
        "score is low, but probing the predicted *change* ranks it first. The change is "
        "small next to `z_t`, so mean-pooling drowns it. The change probe has a true "
        "zero floor and is the dynamics metric used from Phase 2 on.\n\n"
        "Details, failures and reasoning: [`docs/devlog.md`](../docs/devlog.md)."
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
out = Path(__file__).with_name("phase1_frozen_and_dinov2_wm_analysis.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
