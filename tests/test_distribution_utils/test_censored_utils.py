from ..utils import BaseTestClass
import pytest
import numpy as np
import torch
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.LogNormal import CensoredLogNormal
from xgboostlss.distributions.Weibull import CensoredWeibull
from tests.utils import gen_test_data

class TestCensoredUnivariate(BaseTestClass):
    @pytest.fixture(params=[CensoredLogNormal, CensoredWeibull])
    def model(self, request):
        return XGBoostLSS(request.param())

    @pytest.mark.parametrize("weights", [False, True])
    def test_censored_objective_fn_shapes_and_values(self, model, weights):
        predt, lower, upper, *rest = gen_test_data(model, weights=weights, censored=True)
        dmat = rest[-1]
        grad, hess = model.dist.objective_fn(predt, dmat)
        assert isinstance(grad, np.ndarray) and isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape and hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any() and not np.isnan(hess).any()
        assert not np.isinf(grad).any() and not np.isinf(hess).any()

    @pytest.mark.parametrize("loss_fn", ["nll", "crps"])
    @pytest.mark.parametrize("weights", [False, True])
    def test_censored_metric_fn_shapes_and_values(self, model, loss_fn, weights):
        model.dist.loss_fn = loss_fn
        predt, lower, upper, *rest = gen_test_data(model, weights=weights, censored=True)
        dmat = rest[-1]
        name, loss = model.dist.metric_fn(predt, dmat)
        assert name == loss_fn and isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any() and not torch.isinf(loss).any()

    def test_metric_fn_exact_equals_uncensored(self, model):
        predt, labels, *rest = gen_test_data(model, weights=False, censored=False)
        dmat = rest[-1]
        name_c, loss_c = model.dist.metric_fn(predt, dmat)
        underlying_cls = model.dist.__class__.__mro__[2]
        base_model = XGBoostLSS(underlying_cls())
        base_predt, base_labels, *base_rest = gen_test_data(base_model, weights=False, censored=False)
        base_dmat = base_rest[-1]
        name_b, loss_b = base_model.dist.metric_fn(base_predt, base_dmat)
        assert name_c == name_b and torch.allclose(loss_c, loss_b)