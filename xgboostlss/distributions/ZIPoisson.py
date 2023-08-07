from .zero_inflated import ZeroInflatedPoisson as ZeroInflatedPoisson_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class ZIPoisson(DistributionClass):
    """
    Zero-Inflated Poisson distribution class.

    Distributional Parameters
    -------------------------
    rate: torch.Tensor
        Rate parameter of the distribution (often referred to as lambda).
    gate: torch.Tensor
        Probability of extra zeros given via a Bernoulli distribution.

    Source
    -------------------------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L121

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential), "softplus" (softplus) or "relu" (rectified linear unit).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "relu",
                 loss_fn: str = "nll"
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn, "relu": relu_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function for total_count. Please choose from 'exp', 'softplus' or 'relu'.")

        # Set the parameters specific to the distribution
        distribution = ZeroInflatedPoisson_Torch
        param_dict = {"rate": response_fn, "gate": sigmoid_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=True,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )
