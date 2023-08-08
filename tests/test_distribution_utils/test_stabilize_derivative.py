from ..utils import BaseTestClass
import torch


class TestClass(BaseTestClass):
    def test_stabilize_derivative(self, dist_class, stabilization):
        # Create data for testing
        torch.manual_seed(123)
        input_der = torch.rand((10, 1), dtype=torch.float64)

        # Call the function
        stab_der = dist_class.dist.stabilize_derivative(input_der, stabilization)

        # Assertions
        assert isinstance(stab_der, torch.Tensor)
        assert stab_der.shape == input_der.shape
        assert not torch.isnan(stab_der).any()
        assert not torch.isinf(stab_der).any()
        if stabilization == "None":
            assert torch.equal(input_der, stab_der)

    def test_stabilize_derivative_nans(self, dist_class, stabilization):
        # Create data for testing
        torch.manual_seed(123)
        input_der = torch.rand((10, 1), dtype=torch.float64)
        input_der[0] = torch.tensor([float("nan")])

        # Call the function
        stab_der = dist_class.dist.stabilize_derivative(input_der, stabilization)

        # Assertions
        assert isinstance(stab_der, torch.Tensor)
        assert stab_der.shape == input_der.shape
        assert not torch.isnan(stab_der).any()
        assert not torch.isinf(stab_der).any()
