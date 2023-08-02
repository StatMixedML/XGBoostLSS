from xgboostlss.model import *
from ..utils import BaseTestClass


class TestClass(BaseTestClass):

    def test_objective_fn_weights(self, dist_class, loss_fn, stabilization):
        # Create data for testing
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param*4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        dmatrix = xgb.DMatrix(predt, label=labels)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param*4).reshape(-1, dist_class.dist.n_dist_param)
        predt[0, 0] = np.nan
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class.set_base_margin(dmatrix)

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
        np.random.seed(123)
        predt = np.random.rand(dist_class_crps.dist.n_dist_param*4).reshape(-1, dist_class_crps.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        weights = np.ones_like(labels)
        dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
        dist_class_crps.set_base_margin(dmatrix)

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
