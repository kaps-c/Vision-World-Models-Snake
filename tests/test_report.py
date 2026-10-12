from snekwm.eval.report import Panel, plot_curves, probe_table


def test_probe_table_has_one_row_per_model_and_column_per_probe():
    rows = {
        "identity": {"predictor": {"r2_joint": 0.9071}, "change": {"r2_joint": 0.0}},
        "wm": {"predictor": {"r2_joint": 0.9234}},
    }
    md = probe_table(rows)
    lines = md.strip().splitlines()
    assert lines[0] == "| Model | predictor | change |"
    assert lines[2] == "| identity | 0.907 | 0.000 |"
    assert lines[3] == "| wm | 0.923 | — |"


def test_plot_curves_writes_png(tmp_path):
    history = [
        {"epoch": e, "train_loss": 1 / e, "val_mse": 0.5 / e, "identity_mse": 0.2}
        for e in range(1, 4)
    ]
    out = tmp_path / "curves.png"
    panels = [
        Panel("train_loss", "Train loss", log=True),
        Panel("val_mse", "Val MSE", baseline="identity_mse", baseline_label="identity"),
    ]
    plot_curves(history, out, title="run", panels=panels)
    assert out.stat().st_size > 1000
