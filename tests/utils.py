from xgboostlss.model import XGBoostLSS
from xgboostlss import distributions
from xgboostlss.distributions.mixture_distribution_utils import get_component_distributions

import pytest
import importlib
from typing import List
import torch
import numpy as np
import xgboost as xgb


def gen_test_data(dist_class, weights: bool = False):
    """
    Function that generates test data for a given distribution class.

    Arguments:
    ----------
    dist_class (class):
        Distribution class.
    weights (bool):
        Whether to generate weights.

    Returns:
    --------
    predt (np.ndarray):
        Predictions.
    labels (np.ndarray):
        Labels.
    weights (np.ndarray):
        Weights.
    dmatrix (xgb.DMatrix):
        DMatrix.
    """
    if dist_class.dist.univariate:
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        if weights:
            weights = np.ones_like(labels)
            dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
            dist_class.set_base_margin(dmatrix)

            return predt, labels, weights, dmatrix
        else:
            dmatrix = xgb.DMatrix(predt, label=labels)
            dist_class.set_base_margin(dmatrix)

            return predt, labels, dmatrix
    else:
        np.random.seed(123)
        predt = np.random.rand(dist_class.dist.n_dist_param * 4).reshape(-1, dist_class.dist.n_dist_param)
        labels = np.arange(0.1, 0.9, 0.1)
        labels = dist_class.dist.target_append(
            labels,
            dist_class.dist.n_targets,
            dist_class.dist.n_dist_param
        )
        if weights:
            weights = np.ones_like(labels[:, 0], dtype=labels.dtype).reshape(-1, 1)
            dmatrix = xgb.DMatrix(predt, label=labels, weight=weights)
            dist_class.set_base_margin(dmatrix)

            return predt, labels, weights, dmatrix
        else:
            dmatrix = xgb.DMatrix(predt, label=labels)
            dist_class.set_base_margin(dmatrix)

            return predt, labels, dmatrix


def get_distribution_classes(univariate: bool = True,
                             continuous: bool = False,
                             discrete: bool = False,
                             rsample: bool = False,
                             flow: bool = False,
                             expectile: bool = False,
                             mixture: bool = False,
                             ) -> List:
    """
    Function that returns a list of specified distribution classes.

    Arguments:
    ---------
    univariate (bool):
        If True, only return distribution classes that are univariate.
    continuous (bool):
        If True, only return distribution classes that are continuous.
    discrete (bool):
        If True, only return distribution classes that are discrete.
    rsample (bool):
        If True, only return distribution classes that have a rsample method.
    flow (bool):
        If True, only return distribution classes that are Flows.
    expectile (bool):
        If True, only return distribution classes that are Expectiles.
    mixture (bool):
        If True, only return distribution classes that are Mixtures.

    Returns:
    --------
    distribution_classes (List):
        List of all distribution classes in the distributions folder.
    """
    # Get all distribution names
    distns = [dist for dist in dir(distributions) if dist[0].isupper()]

    # Remove specific distributions
    distns_remove = [
        "SplineFlow",
        "Expectile",
        "Mixture"
    ]
    distns = [item for item in distns if item not in distns_remove]

    # Extract all continous univariate distributions
    univar_cont_distns = []
    for distribution_name in distns:
        # Import the module dynamically
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")

        # Get the class dynamically from the module
        distribution_class = getattr(module, distribution_name)

        if distribution_class().univariate and not distribution_class().discrete:
            univar_cont_distns.append(distribution_class)

    # Exctract discrete univariate distributions only
    univar_discrete_distns = []
    for distribution_name in distns:
        # Import the module dynamically
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")

        # Get the class dynamically from the module
        distribution_class = getattr(module, distribution_name)

        if distribution_class().univariate and distribution_class().discrete:
            univar_discrete_distns.append(distribution_class)

    # Extract all multivariate distributions
    multivar_distns = []
    for distribution_name in distns:
        # Import the module dynamically
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")

        # Get the class dynamically from the module
        distribution_class = getattr(module, distribution_name)

        if not distribution_class().univariate:
            multivar_distns.append(distribution_class)

    # Extract distributions only that have a rsample method
    rsample_distns = []
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
                rsample_distns.append(distribution_class)
        except NotImplementedError:
            pass

    if univariate and not flow and not expectile and not mixture:
        if discrete:
            return univar_discrete_distns
        elif rsample:
            return rsample_distns
        elif continuous:
            return univar_cont_distns
        else:
            return univar_cont_distns

    elif not univariate and not flow and not expectile:
        return multivar_distns

    elif flow:
        distribution_name = "SplineFlow"
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")
        # Get the class dynamically from the module
        distribution_class = [getattr(module, distribution_name)]

        return distribution_class

    elif expectile:
        distribution_name = "Expectile"
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")
        # Get the class dynamically from the module
        distribution_class = [getattr(module, distribution_name)]

        return distribution_class

    elif mixture:
        distribution_name = "Mixture"
        module = importlib.import_module(f"xgboostlss.distributions.{distribution_name}")
        # Get the class dynamically from the module
        distribution_class = [getattr(module, distribution_name)]

        return distribution_class


def get_mixture_distribution_classes():
    """
    Function that returns a list of mixture distribution classes, each with a different component distribution.

    Arguments:
    ---------
    None

    Returns:
    --------
    mixt_dist (List):
        List of all mixture distribution classes.
    """
    mix_class = get_distribution_classes(mixture=True)[0]
    comp_dists = get_component_distributions()
    mixt_dist = []
    for dist in comp_dists:
        module = importlib.import_module(f"xgboostlss.distributions.{dist}")
        comp_dist = getattr(module, dist)
        mixt_dist.append(XGBoostLSS(mix_class(comp_dist())))

    return mixt_dist


class BaseTestClass:
    @pytest.fixture(params=get_distribution_classes(continuous=True))
    def univariate_cont_dist(self, request):
        return request.param

    @pytest.fixture(params=get_distribution_classes(discrete=True))
    def univariate_discrete_dist(self, request):
        return request.param

    @pytest.fixture(params=get_distribution_classes(univariate=False))
    def multivariate_dist(self, request):
        return request.param

    @pytest.fixture(params=get_distribution_classes(flow=True))
    def flow_dist(self, request):
        return request.param

    @pytest.fixture(params=get_distribution_classes(expectile=True))
    def expectile_dist(self, request):
        return request.param

    @pytest.fixture(params=get_distribution_classes(mixture=True))
    def mixture_dist(self, request):
        return request.param

    @pytest.fixture(params=
                    get_distribution_classes() +
                    get_distribution_classes(discrete=True) +
                    get_distribution_classes(expectile=True) +
                    get_distribution_classes(flow=True) +
                    get_distribution_classes(univariate=False)
                    )
    def dist_class(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=get_distribution_classes(flow=True))
    def flow_class(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=get_distribution_classes(univariate=False))
    def multivariate_class(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=get_distribution_classes(rsample=True))
    def dist_class_crps(self, request):
        return XGBoostLSS(request.param())

    @pytest.fixture(params=get_mixture_distribution_classes())
    def mixture_class(self, request):
        return request.param

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

    @pytest.fixture(params=["individual", "grouped"])
    def hessian_mode(self, request):
        return request.param
