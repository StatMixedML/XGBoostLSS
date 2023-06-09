from torch.distributions import NegativeBinomial as NegativeBinomial_Torch
from xgboostlss.utils import *
from .distribution_utils import *


class NegativeBinomial:
    """
    NegativeBinomial distribution class.

    Distributional Parameters
    -------------------------
    total_count: torch.Tensor
        non-negative number of negative Bernoulli trials to stop,  although the distribution is still valid for real valued count
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
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "exp", "softplus" or "relu".
    response_fn_probs: str
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "sigmoid".
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
        # Specify Response and Link Functions for total_count
        if response_fn_total_count == "exp":
            response_fn_total_count = exp_fn
            inverse_response_fn_total_count = log_fn
        elif response_fn_total_count == "softplus":
            response_fn_total_count = softplus_fn
            inverse_response_fn_total_count = softplusinv_fn
        elif response_fn_total_count == "relu":
            response_fn_total_count = relu_fn
            inverse_response_fn_total_count = reluinv_fn
        else:
            raise ValueError("Invalid response function for total_count. Please choose from 'exp', 'softplus' or relu.")

        # Specify Response and Link Functions for probs
        if response_fn_probs == "sigmoid":
            response_fn_probs = sigmoid_fn
            inverse_response_fn_probs = sigmoidinv_fn
        else:
            raise ValueError("Invalid response function for probs. Please select 'sigmoid'.")

        param_dict = {"total_count": response_fn_total_count, "probs": response_fn_probs}
        param_dict_inv = {"total_count": inverse_response_fn_total_count, "probs": inverse_response_fn_probs}
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=NegativeBinomial_Torch,
                                            univariate=True,
                                            discrete=True,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names,
                                            loss_fn=loss_fn
                                            )
