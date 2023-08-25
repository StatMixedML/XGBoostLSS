from ..utils import BaseTestClass
from xgboostlss.distributions.mixture_distribution_utils import get_component_distributions


class TestClass(BaseTestClass):
    def test_create_spline_flow(self):
        comp_dists = get_component_distributions()

        # Assertions
        assert isinstance(comp_dists, list)
        assert all(isinstance(dist, str) for dist in comp_dists)
