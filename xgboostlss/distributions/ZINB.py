from .zero_inflated import ZeroInflatedNegativeBinomial as ZeroInflatedNegativeBinomial_Torch
from xgboostlss.utils import *
from .distribution_utils import *


class ZINB:
    """
    Zero-Inflated Negative Binomial distribution class.

    Distributional Parameters
    -------------------------
    total_count: torch.Tensor
        Non-negative number of negative Bernoulli trials to stop, although the distribution is still valid for real valued count.
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
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "exp", "softplus" or "relu".
    response_fn_probs: str
        When a custom objective and metric are provided, XGBoost doesn't know its response and link function. Hence,
        the user is responsible for specifying the transformations. Options are "sigmoid".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
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

        param_dict = {"total_count": response_fn_total_count,
                      "probs": response_fn_probs,
                      "gate": sigmoid_fn
                      }
        param_dict_inv = {"total_count": inverse_response_fn_total_count,
                          "probs": inverse_response_fn_probs,
                          "gate": sigmoidinv_fn
                          }
        distribution_arg_names = list(param_dict.keys())

        # Specify Distribution
        self.dist_class = DistributionClass(distribution=ZeroInflatedNegativeBinomial_Torch,
                                            univariate=True,
                                            discrete=True,
                                            n_dist_param=len(param_dict),
                                            stabilization=stabilization,
                                            param_dict=param_dict,
                                            param_dict_inv=param_dict_inv,
                                            distribution_arg_names=distribution_arg_names,
                                            loss_fn=loss_fn
                                            )
