import pytest
from xgboostlss.distributions.Laplace import *


@pytest.fixture
def dist():
    return Laplace()


def test_init(dist):
    assert isinstance(dist.stabilization, str)
    assert dist.stabilization is not None
    assert isinstance(dist.loss_fn, str)
    assert dist.loss_fn is not None


def test_distribution_parameters(dist):
    assert isinstance(dist.param_dict, dict)
    assert set(dist.param_dict.keys()) == set(dist.distribution_arg_names)
    assert all(callable(func) for func in dist.param_dict.values())
    assert dist.n_dist_param == len(dist.distribution_arg_names)
    assert isinstance(dist.n_dist_param, int)
    assert isinstance(dist.distribution_arg_names, list)
    assert dist.distribution_arg_names == list(dist.distribution.arg_constraints.keys())


def test_defaults(dist):
    assert isinstance(dist.univariate, bool)
    assert dist.univariate is True
    assert isinstance(dist.discrete, bool)
    assert dist.discrete is False
    assert dist.tau is None
    assert isinstance(dist.penalize_crossing, bool)
