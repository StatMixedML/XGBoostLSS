from torch.distributions import Beta as Beta_Torch
from xgboostlss.utils import *
from .distribution_utils import *


class Beta:
    """
    Beta distribution class.

    Distributional Parameters
    -------------------------
    concentration1: torch.Tensor
        1st concentration parameter of the distribution (often referred to as alpha).
    concentration0: torch.Tensor
        2nd concentration parameter of the distribution (often referred to as beta).

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#beta

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "exp" or "softplus".
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp"
                 ):
        # Check Response Function
        if response_fn == "exp":
            response_fn = exp_fn
            inverse_response_fn = log_fn
        elif response_fn == "softplus":
            response_fn = softplus_fn
            inverse_response_fn = softplusinv_fn
        else:
            raise ValueError("Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Specify Response and Link Functions
        param_dict = {"concentration1": response_fn, "concentration0": response_fn}
        param_dict_inv = {"concentration1": inverse_response_fn, "concentration0": inverse_response_fn}
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=Beta_Torch,
                                            univariate=True,
                                            discrete=False,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names
                                            )
