from torch.distributions import Cauchy as Cauchy_Torch
from xgboostlss.utils import *
from .distribution_utils import *

class Cauchy:
    """
    Cauchy distribution class.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mode or median of the distribution.
    scale: torch.Tensor
        Half width at half maximum.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#cauchy

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "exp" or "softplus".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll"
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
        param_dict = {"loc": identity_fn, "scale": response_fn}
        param_dict_inv = {"loc": identity_fn, "scale": inverse_response_fn}
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=Cauchy_Torch,
                                            univariate=True,
                                            discrete=False,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names,
                                            loss_fn=loss_fn
                                            )
