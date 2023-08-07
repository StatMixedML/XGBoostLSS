import pytest
import torch
from xgboostlss import utils


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
        predt = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Call the function
        predt_transformed = response_fn(predt)

        # Assertions
        assert isinstance(predt_transformed, torch.Tensor)
        assert not torch.isnan(predt_transformed).any()
        assert not torch.isinf(predt_transformed).any()
