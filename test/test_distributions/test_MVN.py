import pytest
from xgboostlss.distributions.MVN import *


@pytest.fixture
def dist():
    return MVN(D=2)


def test_init(dist):
    assert isinstance(dist.n_targets, int)
    assert dist.n_targets > 0
    assert isinstance(dist.stabilization, str)
    assert dist.stabilization is not None
    assert isinstance(dist.loss_fn, str)
    assert dist.loss_fn is not None


def test_distribution_parameters(dist):
    assert isinstance(dist.param_dict, dict)
    assert all(callable(func) for func in dist.param_dict.values())
    assert isinstance(dist.n_dist_param, int)
    assert isinstance(dist.distribution_arg_names, list)
    assert dist.param_transform is not None
    assert callable(dist.param_transform) is True
    assert dist.create_param_dict is not None
    assert callable(dist.create_param_dict) is True
    assert dist.get_dist_params is not None
    assert callable(dist.get_dist_params) is True


def test_defaults(dist):
    assert isinstance(dist.univariate, bool)
    assert dist.univariate is False
    assert isinstance(dist.discrete, bool)
    assert dist.discrete is False
    assert dist.rank is None
