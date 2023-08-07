from ..utils import BaseTestClass
from pyro.distributions import TransformedDistribution


class TestClass(BaseTestClass):
    def test_create_spline_flow(self, flow_class):
        # Create normalizing flow
        gen_flow = flow_class.dist.create_spline_flow(input_dim=1)

        # Assertions
        assert isinstance(gen_flow, TransformedDistribution)
