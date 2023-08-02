from ..utils import BaseTestClass
import pandas as pd


class TestClass(BaseTestClass):
    def test_draw_samples(self, dist_class, loss_fn):
        # Create data for testing
        predt_params = pd.DataFrame([0.5 for _ in range(dist_class.dist.n_dist_param)]).T

        # Call the function
        dist_samples = dist_class.dist.draw_samples(predt_params)

        # Assertions
        assert isinstance(dist_samples, (pd.DataFrame, type(None)))
