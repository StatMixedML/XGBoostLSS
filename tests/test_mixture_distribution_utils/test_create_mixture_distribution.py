from ..utils import BaseTestClass
import torch

class TestClass(BaseTestClass):
    def test_create_spline_flow(self, mixture_class):
        # Create params for testing
        torch.manual_seed(123)
        params = torch.rand(mixture_class.dist.n_dist_param).reshape(1, -1)
        params = torch.split(params, mixture_class.dist.M, dim=1)

        # Transform parameters to response scale
        params = [response_fn(params[i]) for i, response_fn in enumerate(mixture_class.dist.param_dict.values())]

        # Call Function
        dist_fit = mixture_class.dist.create_mixture_distribution(params)


        # Assertions
        assert isinstance(dist_fit, torch.distributions.mixture_same_family.MixtureSameFamily)
        assert dist_fit._num_component == mixture_class.dist.M