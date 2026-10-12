import pytest
import torch

from snekwm.eval.collapse import CollapseError, check_rank_ratio, effective_rank


def test_effective_rank_of_isotropic_data_is_close_to_dimension():
    torch.manual_seed(0)
    assert effective_rank(torch.randn(4000, 16)) > 15


def test_effective_rank_of_rank_one_data_is_one():
    x = torch.randn(500, 1) @ torch.randn(1, 16)
    assert effective_rank(x) == pytest.approx(1.0, abs=0.05)


def test_effective_rank_ignores_the_mean():
    torch.manual_seed(0)
    x = torch.randn(2000, 8)
    assert effective_rank(x + 100) == pytest.approx(effective_rank(x), rel=1e-3)


def test_rank_ratio_returns_ratio_when_healthy():
    torch.manual_seed(0)
    online, target = torch.randn(1000, 8), torch.randn(1000, 8)
    assert check_rank_ratio(online, target, minimum=0.5) == pytest.approx(1.0, abs=0.1)


def test_rank_ratio_raises_on_dimensional_collapse():
    torch.manual_seed(0)
    collapsed = torch.randn(1000, 1) @ torch.randn(1, 8)
    with pytest.raises(CollapseError) as err:
        check_rank_ratio(collapsed, torch.randn(1000, 8), minimum=0.5)
    assert err.value.ratio < 0.5
