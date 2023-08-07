from ..utils import BaseTestClass
import pandas as pd
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_draw_samples(self, dist_class, loss_fn):
        if dist_class.dist.univariate:
            # Create data for testing
            predt_params = pd.DataFrame(np.array([0.5 for _ in range(dist_class.dist.n_dist_param)], dtype="float32")).T

            # Call the function
            dist_samples = dist_class.dist.draw_samples(predt_params)

        else:
            # Create data for testing
            n_obs = 1
            predt = np.array([0.5 for _ in range(dist_class.dist.n_dist_param)])
            predt = predt.reshape(-1, dist_class.dist.n_dist_param)
            predt = [
                torch.tensor(predt[:, i].reshape(-1, 1), requires_grad=False) for i in
                range(dist_class.dist.n_dist_param)
            ]
            predt_transformed = dist_class.dist.param_transform(predt, dist_class.dist.param_dict,
                                                                dist_class.dist.n_targets, rank=dist_class.dist.rank,
                                                                n_obs=n_obs)

            # Call the function
            if dist_class.dist.distribution.__name__ == "Dirichlet":
                dist_kwargs = dict(zip(dist_class.dist.distribution_arg_names, [predt_transformed]))
            else:
                dist_kwargs = dict(zip(dist_class.dist.distribution_arg_names, predt_transformed))
            dist_pred = dist_class.dist.distribution(**dist_kwargs)
            dist_samples = dist_class.dist.draw_samples(dist_pred)

            # Assertions
            assert isinstance(dist_samples, (pd.DataFrame, type(None)))
            assert not dist_samples.isna().any().any()
            assert not np.isinf(dist_samples.iloc[:, 1:]).any().any()

