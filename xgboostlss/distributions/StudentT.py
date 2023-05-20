from torch.distributions import StudentT as StudentT_Torch
from xgboostlss.utils import *
from .distribution_utils import *


class StudentT:
    """
    Student-T Distribution Class

    Distributional Parameters
    -------------------------
    df: torch.Tensor
        Degrees of freedom.
    loc: torch.Tensor
        Mean of the distribution.
    scale: torch.Tensor
        Scale of the distribution.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#studentt
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
        param_dict = {"df": lambda x: response_fn(x) + torch.tensor(2.0),
                      "loc": identity_fn,
                      "scale": response_fn
                      }
        param_dict_inv = {"df": inverse_response_fn,
                          "loc": identity_fn,
                          "scale": inverse_response_fn
                          }
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=StudentT_Torch,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names
                                            )
