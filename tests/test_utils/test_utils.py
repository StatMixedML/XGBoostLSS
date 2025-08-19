import pytest
import torch
from xgboostlss import utils
import numpy as np


def get_response_fn():
    functions_list = [fn for fn in dir(utils) if "_fn" in fn]

    func_list = []
    for func_name in functions_list:
        func_list.append(getattr(utils, func_name))

    return func_list


class TestClass:
    @pytest.fixture(params=get_response_fn())
    def response_fn(self, request):
        return request.param

    def test_response_fn(self, response_fn):
        # Create Data for testing
        predt = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(-1, 1)

        # Call the function
        predt_transformed = response_fn(predt)

        # Assertions
        assert isinstance(predt_transformed, torch.Tensor)
        assert not torch.isnan(predt_transformed).any()
        assert not torch.isinf(predt_transformed).any()

    def test_response_inverse_fn(self, response_fn):
        # Create Data for testing
        predt = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(-1, 1)
        predt_transformed = response_fn(predt)

        response_inv_fn = utils.INVERSE_LOOKUP.get(response_fn.__name__, None)
        if response_inv_fn:
            inverse_predt = response_inv_fn(predt_transformed)

            np.testing.assert_allclose(
                inverse_predt.numpy(), predt.numpy(), atol=utils._EPS
            )
