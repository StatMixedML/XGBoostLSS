import torch
from torch.distributions import identity_transform, SigmoidTransform, SoftplusTransform
from pyro.distributions import Normal
from pyro.distributions.transforms import Spline
from .flow_utils import NormalizingFlowClass
from ..utils import identity_fn


class SplineFlow(NormalizingFlowClass):
    """
    Spline Flow class.

    The spline flow is a normalizing flow based on element-wise rational spline bijections of linear and quadratic
    order (Durkan et al., 2019; Dolatabadi et al., 2020). Rational splines are functions that are comprised of segments
    that are the ratio of two polynomials. Rational splines offer an excellent combination of functional flexibility
    whilst maintaining a numerically stable inverse.

    For more details, see:
    - Durkan, C., Bekasov, A., Murray, I. and Papamakarios, G. Neural Spline Flows. NeurIPS 2019.
    - Dolatabadi, H. M., Erfani, S. and Leckie, C., Invertible Generative Modeling using Linear Rational Splines. AISTATS 2020.


    Source
    ---------
    https://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.transforms.Spline


    Arguments
    ---------
    target_support: str
        The target support. Options are
            - "real": [-inf, inf]
            - "positive": [0, inf]
            - "positive_integer": [0, 1, 2, 3, ...]
            - "unit_interval": [0, 1]
    count_bins: int
        The number of segments comprising the spline.
    bound: float
        The quantity "K" determining the bounding box, [-K,K] x [-K,K] of the spline. By adjusting the
        "K" value, you can control the size of the bounding box and consequently control the range of inputs that
        the spline transform operates on. Larger values of "K" will result in a wider valid range for the spline
        transformation, while smaller values will restrict the valid range to a smaller region. Should be chosen
        based on the range of the data.
    order: str
        The order of the spline. Options are "linear" or "quadratic".
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD" or "L2".
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    """
    def __init__(self,
                 target_support: str = "real",
                 count_bins: int = 8,
                 bound: float = 3.0,
                 order: str = "linear",
                 stabilization: str = "None",
                 loss_fn: str = "nll"
                 ):

        # Check if stabilization method is valid.
        if not isinstance(stabilization, str):
            raise ValueError("stabilization must be a string.")
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Options are 'None', 'MAD' or 'L2'.")

        # Check if loss function is valid.
        if not isinstance(loss_fn, str):
            raise ValueError("loss_fn must be a string.")
        if loss_fn not in ["nll", "crps"]:
            raise ValueError("Invalid loss_fn. Options are 'nll' or 'crps'.")

        # Number of parameters
        if not isinstance(order, str):
            raise ValueError("order must be a string.")
        if order == "quadratic":
            n_params = 2*count_bins + (count_bins-1)
        elif order == "linear":
            n_params = 3*count_bins + (count_bins-1)
        else:
            raise ValueError("Invalid order specification. Options are 'linear' or 'quadratic'.")

        # Specify Target Transform
        if not isinstance(target_support, str):
            raise ValueError("target_support must be a string.")
        if target_support == "real":
            target_transform = identity_transform
            discrete = False
        elif target_support == "positive":
            target_transform = SoftplusTransform()
            discrete = False
        elif target_support == "positive_integer":
            target_transform = SoftplusTransform()
            discrete = True
        elif target_support == "unit_interval":
            target_transform = SigmoidTransform()
            discrete = False
        else:
            raise ValueError("Invalid target_support. Options are 'real', 'positive', 'positive_integer' or 'unit_interval'.")

        # Check if count_bins is valid
        if not isinstance(count_bins, int):
            raise ValueError("count_bins must be an integer.")
        if count_bins <= 0:
            raise ValueError("count_bins must be a positive integer > 0.")

        # Check if bound is float
        if not isinstance(bound, float):
            raise ValueError("bound must be a float.")

        # Specify parameter dictionary
        param_dict = {f"param_{i + 1}": identity_fn for i in range(n_params)}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Normalizing Flow Class
        super().__init__(base_dist=Normal,                     # Base distribution, currently only Normal is supported.
                         flow_transform=Spline,
                         count_bins=count_bins,
                         bound=bound,
                         order=order,
                         n_dist_param=n_params,
                         param_dict=param_dict,
                         target_transform=target_transform,
                         discrete=discrete,
                         univariate=True,
                         stabilization=stabilization,
                         loss_fn=loss_fn
                         )
