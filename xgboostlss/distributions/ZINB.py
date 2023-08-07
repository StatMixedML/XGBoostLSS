from .zero_inflated import ZeroInflatedNegativeBinomial as ZeroInflatedNegativeBinomial_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class ZINB(DistributionClass):
    """
    Zero-Inflated Negative Binomial distribution class.

    Distributional Parameters
    -------------------------
    total_count: torch.Tensor
        Non-negative number of negative Bernoulli trials to stop.
    probs: torch.Tensor
        Event probabilities of success in the half open interval [0, 1).
    gate: torch.Tensor
        Probability of extra zeros given via a Bernoulli distribution.

    Source
    -------------------------
    https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/zero_inflated.py#L150

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn_total_count: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential), "softplus" (softplus) or "relu" (rectified linear unit).
    response_fn_probs: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "sigmoid" (sigmoid).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn_total_count: str = "relu",
                 response_fn_probs: str = "sigmoid",
                 loss_fn: str = "nll"
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        #  Specify Response Functions for total_count
        response_functions_total_count = {"exp": exp_fn, "softplus": softplus_fn, "relu": relu_fn}
        if response_fn_total_count in response_functions_total_count:
            response_fn_total_count = response_functions_total_count[response_fn_total_count]
        else:
            raise ValueError(
                "Invalid response function for total_count. Please choose from 'exp', 'softplus' or 'relu'.")

        #  Specify Response Functions for probs
        response_functions_probs = {"sigmoid": sigmoid_fn}
        if response_fn_probs in response_functions_probs:
            response_fn_probs = response_functions_probs[response_fn_probs]
        else:
            raise ValueError(
                "Invalid response function for probs. Please select 'sigmoid'.")

        # Set the parameters specific to the distribution
        distribution = ZeroInflatedNegativeBinomial_Torch
        param_dict = {"total_count": response_fn_total_count, "probs": response_fn_probs, "gate": sigmoid_fn}
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
