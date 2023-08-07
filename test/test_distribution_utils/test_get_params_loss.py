from ..utils import BaseTestClass, gen_test_data
from typing import List
import numpy as np
import torch


class TestClass(BaseTestClass):
    def test_get_params_loss(self, dist_class, loss_fn, requires_grad):
        # Create data for testing
        predt, target, _ = gen_test_data(dist_class)
        if dist_class.dist.univariate:
            target = torch.tensor(target)
        else:
            target = torch.tensor(target)[:, :dist_class.dist.n_targets]
        start_values = np.array([0.5 for _ in range(dist_class.dist.n_dist_param)])

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_get_params_loss_nans(self, dist_class, loss_fn, requires_grad):
        # Create data for testing
        predt, target, _ = gen_test_data(dist_class)
        predt[0, 0] = np.nan
        if dist_class.dist.univariate:
            target = torch.tensor(target)
        else:
            target = torch.tensor(target)[:, :dist_class.dist.n_targets]
        start_values = np.array([0.5 for _ in range(dist_class.dist.n_dist_param)])

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Call the function
        predt, loss = dist_class.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_get_params_loss_crps(self, dist_class_crps, requires_grad):
        # Create data for testing
        predt, target, _ = gen_test_data(dist_class_crps)
        if dist_class_crps.dist.univariate:
            target = torch.tensor(target)
        else:
            target = torch.tensor(target)[:, :dist_class_crps.dist.n_targets]
        start_values = np.array([0.5 for _ in range(dist_class_crps.dist.n_dist_param)])

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Call the function
        predt, loss = dist_class_crps.dist.get_params_loss(predt, target, start_values, requires_grad)

        # Assertions
        assert isinstance(predt, List)
        for i in range(len(predt)):
            assert isinstance(predt[i], torch.Tensor)
            assert not torch.isnan(predt[i]).any()
            assert not torch.isinf(predt[i]).any()
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
