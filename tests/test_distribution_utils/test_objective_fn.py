from ..utils import BaseTestClass, gen_test_data
import numpy as np


class TestClass(BaseTestClass):
    def test_objective_fn_weights(self, dist_class, loss_fn, stabilization):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(dist_class, weights=True)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        dist_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = dist_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_no_weights(self, dist_class, loss_fn, stabilization):
        # Create data for testing
        predt, labels, dmatrix = gen_test_data(dist_class, weights=False)

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        dist_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = dist_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_nans(self, dist_class, loss_fn, stabilization):
        # Create data for testing and et some predt to nan
        predt, labels, weights, dmatrix = gen_test_data(dist_class, weights=True)
        predt[0, 0] = np.nan

        # Set the loss function for testing
        dist_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        dist_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = dist_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_crps(self, dist_class_crps, stabilization):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(dist_class_crps, weights=True)

        # Set the loss function for testing
        dist_class_crps.dist.loss_fn = "crps"

        # Set the stabilization for testing
        dist_class_crps.dist.stabilization = stabilization

        # Call the function
        grad, hess = dist_class_crps.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_mixture_weights(self, mixture_class, loss_fn, stabilization):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(mixture_class, weights=True)

        # Set the loss function for testing
        mixture_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        mixture_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = mixture_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_mixture_no_weights(self, mixture_class, loss_fn, stabilization):
        # Create data for testing
        predt, labels, dmatrix = gen_test_data(mixture_class, weights=False)

        # Set the loss function for testing
        mixture_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        mixture_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = mixture_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_mixture_nans(self, mixture_class, loss_fn, stabilization):
        # Create data for testing
        predt, labels, weights, dmatrix = gen_test_data(mixture_class, weights=True)
        predt[0, 0] = np.nan

        # Set the loss function for testing
        mixture_class.dist.loss_fn = loss_fn

        # Set the stabilization for testing
        mixture_class.dist.stabilization = stabilization

        # Call the function
        grad, hess = mixture_class.dist.objective_fn(predt, dmatrix)

        # Assertions
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == predt.flatten().shape
        assert hess.shape == predt.flatten().shape
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()
