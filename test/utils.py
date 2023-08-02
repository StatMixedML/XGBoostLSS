from xgboostlss.model import XGBoostLSS
from xgboostlss import distributions
import pytest
import importlib
from typing import List
import torch


def get_distribution_classes(has_rsample: bool = False) -> List:
    """
    Function that returns a list of all distribution classes in the distributions folder.

    Arguments:
    ---------
    has_rsample (bool):
        If True, only return distribution classes that have a rsample method.

    Returns:
    --------
    distribution_classes (List):
        List of all distribution classes in the distributions folder.
    """
    # Get all distribution names
    distns = [dist for dist in dir(distributions) if dist[0].isupper()]

    # Remove SplineFlow from distns
    distns.remove("SplineFlow")

    # Loop through each distribution name and import the corresponding class
    distribution_classes = []

    # Extract all univariate distributions
    if not has_rsample:
        for distribution_name in distns:
            # Import the module dynamically
            module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")

            # Get the class dynamically from the module
            distribution_class = getattr(module, distribution_name)

            # Add the class object to the list if it is a univariate distribution
            if distribution_class().univariate:
                distribution_classes.append(distribution_class)

    # Extract distributions only that have a rsample method
    else:
        for distribution_name in distns:
            # Import the module dynamically
            module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")

            # Get the class dynamically from the module
            distribution_class = getattr(module, distribution_name)

            # Create an instance of the distribution class
            dist_class = XGBoostLSS(distribution_class())
            params = torch.tensor([0.5 for _ in range(dist_class.dist.n_dist_param)])

            # Check if the distribution is univariate and has a rsample method
            if distribution_class().univariate and dist_class.dist.tau is None:
                dist_kwargs = dict(zip(dist_class.dist.distribution_arg_names, params))
                dist_fit = dist_class.dist.distribution(**dist_kwargs)

            elif distribution_class().univariate and dist_class.dist.tau is not None:
                dist_fit = dist_class.dist.distribution(params)

            try:
                dist_fit.rsample()
                if distribution_class().univariate:
                    distribution_classes.append(distribution_class)
            except NotImplementedError:
                pass

    return distribution_classes


class BaseTestClass:
    @pytest.fixture(params=get_distribution_classes())
    def dist_class(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=get_distribution_classes(has_rsample=True))
    def dist_class_crps(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=["nll"])
    def loss_fn(self, request):
        return request.param

    @pytest.fixture(params=["None", "MAD", "L2"])
    def stabilization(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def requires_grad(self, request):
        return request.param

    @pytest.fixture(params=["samples", "quantiles", "parameters", "expectiles"])
    def pred_type(self, request):
        return request.param
