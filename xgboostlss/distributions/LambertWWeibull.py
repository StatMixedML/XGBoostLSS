import torchlambertw.distributions as tlwd
from .distribution_utils import DistributionClass
from ..utils import *


class TailLambertWWeibull(DistributionClass):
    """
    Tail Lambert W x Weibull distribution class.

    Distributional Parameters
    -------------------------
    concentration: torch.Tensor
        Concentration of the distribution (often referred as the shape parameter).
    scale: torch.Tensor
        Scale parameter of the Weibull distribution.
    tailweight: torch.Tensor:
        Tail-weight of the distribution (often referred to as delta or h).

    Source
    -------------------------
    https://www.github.com/gmgeorg/torchlambertw and https://www.github.com/gmgeorg/pylambertw

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """

    def __init__(
        self,
        stabilization: str = "None",
        response_fn: str = "softplus",
        loss_fn: str = "nll",
    ):
        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll", "crps"]:
            raise ValueError(
                "Invalid loss function. Please choose from 'nll' or 'crps'."
            )

        # Specify Response Functions
        response_functions = {
            # For (concentation, scale, tailweight)
            "exp": (exp_fn, exp_fn, exp_fn),
            "softplus": (softplus_fn, softplus_fn, softplus_fn),
        }
        if response_fn in response_functions:
            (
                response_fn_concentration,
                response_fn_scale,
                response_fn_tailweight,
            ) = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'."
            )

        # Set the parameters specific to the distribution
        distribution = tlwd.TailLambertWWeibull
        param_dict = {
            "scale": response_fn_scale,
            "concentration": response_fn_concentration,
            "tailweight": response_fn_tailweight,
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
        )


class SkewLambertWWeibull(DistributionClass):
    """
    Skew Lambert W x Weibull distribution class.

    Distributional Parameters
    -------------------------
    concentration: torch.Tensor
        Concentration of the distribution (often referred as the shape parameter).
    scale: torch.Tensor
        Scale parameter of the Weibull distribution.
    skewweight: torch.Tensor:
        Skew-weight of the distribution (also referred to as gamma).

    Source
    -------------------------
    https://www.github.com/gmgeorg/torchlambertw and https://www.github.com/gmgeorg/pylambertw

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """

    def __init__(
        self,
        stabilization: str = "None",
        response_fn: str = "softplus",
        loss_fn: str = "nll",
    ):
        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll", "crps"]:
            raise ValueError(
                "Invalid loss function. Please choose from 'nll' or 'crps'."
            )

        # Specify Response Functions
        response_functions = {
            # For (concentation, scale, tailweight)
            "exp": (exp_fn, exp_fn, exp_fn),
            "softplus": (softplus_fn, softplus_fn, softplus_fn),
        }
        if response_fn in response_functions:
            (
                response_fn_concentration,
                response_fn_scale,
                response_fn_skewweight,
            ) = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'."
            )

        # Set the parameters specific to the distribution
        distribution = tlwd.SkewLambertWWeibull
        param_dict = {
            "scale": response_fn_scale,
            "concentration": response_fn_concentration,
            "skewweight": response_fn_skewweight,
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
        )
