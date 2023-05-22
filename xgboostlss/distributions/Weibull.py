from torch.distributions import Weibull as Weibull_Torch
from xgboostlss.utils import *
from .distribution_utils import *


class Weibull:
    """
    Weibull distribution class.

    Distributional Parameters
    -------------------------
    scale: torch.Tensor
        Scale parameter of distribution (lambda).
    concentration: torch.Tensor
        Concentration parameter of distribution (k/shape).

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#weibull
    """
    def __init__(self,
                 stabilization: str,
                 response_fn: str = "exp"
                 ):

        # When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        # the user is responsible for specifying the transformations.

        if response_fn == "exp":
            response_fn = exp_fn
            inverse_response_fn = log_fn
        elif response_fn == "softplus":
            response_fn = softplus_fn
            inverse_response_fn = softplusinv_fn
        else:
            raise ValueError("Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Specify Response and Link Functions
        param_dict = {"scale": response_fn, "concentration": response_fn}
        param_dict_inv = {"scale": inverse_response_fn, "concentration": inverse_response_fn}
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=Weibull_Torch,
                                            univariate=True,
                                            discrete=False,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names
                                            )
