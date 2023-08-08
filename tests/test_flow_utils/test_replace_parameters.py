from ..utils import BaseTestClass
from pyro.distributions import TransformedDistribution
from typing import List
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_replace_parameters(self, flow_class):
        # Specify Normalizing Flow
        predt = np.array([0.5 for _ in range(flow_class.dist.n_dist_param)]).reshape(-1, 1).T
        predt = torch.tensor(predt, dtype=torch.float32)
        flow_dist = flow_class.dist.create_spline_flow(input_dim=1)

        # Cal the function
        params, flow_dist = flow_class.dist.replace_parameters(predt, flow_dist)

        # Assertions
        assert isinstance(flow_dist, TransformedDistribution)
        assert isinstance(params, List)
        for i in range(len(params)):
            assert isinstance(params[i], torch.Tensor)
            assert not torch.isnan(params[i]).any()
            assert not torch.isinf(params[i]).any()
