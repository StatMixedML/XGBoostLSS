from torch.distributions import NegativeBinomial as NegativeBinomial_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class NegativeBinomial(DistributionClass):
    """
    NegativeBinomial distribution class.

    Distributional Parameters
    -------------------------
    total_count: torch.Tensor
        Non-negative number of negative Bernoulli trials to stop.
    probs: torch.Tensor
        Event probabilities of success in the half open interval [0, 1).
    logits: torch.Tensor
        Event log-odds for probabilities of success.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#negativebinomial

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
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn_total_count: str = "relu",
                 response_fn_probs: str = "sigmoid",
                 loss_fn: str = "nll"
                 ):
        #  Specify Response Functions for total_count
        if response_fn_total_count == "exp":
            response_fn_total_count = exp_fn
        elif response_fn_total_count == "softplus":
            response_fn_total_count = softplus_fn
        elif response_fn_total_count == "relu":
            response_fn_total_count = relu_fn
        else:
            raise ValueError("Invalid response function for total_count. Please choose from 'exp', 'softplus' or relu.")

        #  Specify Response Functions for probs
        if response_fn_probs == "sigmoid":
            response_fn_probs = sigmoid_fn
        else:
            raise ValueError("Invalid response function for probs. Please select 'sigmoid'.")

        # Set the parameters specific to the distribution
        distribution = NegativeBinomial_Torch
        param_dict = {"total_count": response_fn_total_count, "probs": response_fn_probs}
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
