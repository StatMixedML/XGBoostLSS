from .mixture_distribution_utils import get_component_distributions, MixtureDistributionClass
from ..utils import *
from functools import partial


class Mixture(MixtureDistributionClass):
    """
    Mixture-Density distribution class.

    Implements a mixture-density distribution for univariate targets, where all components are from different
    parameterizations of the same distribution-type. A mixture-density distribution is a concept used to model a
    complex distribution that arises from combining multiple simpler distributions. The Mixture-Density distribution
    is parameterized by a categorical selecting distribution (over M components) and M-component distributions. For more
    information on the Mixture-Density distribution, see:

        Bishop, C. M. (1994). Mixture density networks. Technical Report NCRG/4288, Aston University, Birmingham, UK.
    

    Distributional Parameters
    -------------------------
    Inherits the distributional parameters from the component distributions.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#mixturesamefamily

    Parameters
    -------------------------
    component_distribution: torch.distributions.Distribution
        Distribution class for the components of the mixture distribution. Has to be one of the available
        univariate distributions of the package.
    M: int
        Number of components in the mixture distribution.
    hessian_mode: str
        Mode for computing the Hessian. Must be one of the following:

            - "individual": Each parameter is treated as a separate tensor. As a result, the Hessian corresponds to the
            second-order derivative with respect to that specific parameter only. The resulting Hessians capture the
            curvature of the loss w.r.t. each individual parameter. This is usually more runtime intensive, but can
            be more accurate.

            - "grouped": Each tensor contains all parameters for a specific parameter-type, e.g., for a Gaussian-Mixture
            with M=2, loc=[loc_1, loc_2], scale=[scale_1, scale_2], and mix_prob=[mix_prob_1, mix_prob_2]. When
            computing the Hessian, the derivatives for all parameters in the respective tensor are calculated jointly.
            The resulting Hessians capture the curvature of the loss w.r.t. the entire parameter-type. This is usually
            less runtime intensive, but can be less accurate.
    tau: float, non-negative scalar temperature.
        The Gumbel-softmax distribution is a continuous distribution over the simplex, which can be thought of as a "soft"
        version of a categorical distribution. It’s a way to draw samples from a categorical distribution in a
        differentiable way. The motivation behind using the Gumbel-Softmax is to make the discrete sampling process of
        categorical variables differentiable, which is useful in gradient-based optimization problems. To sample from a
        Gumbel-Softmax distribution, one would use the Gumbel-max trick: add a Gumbel noise to logits and apply the softmax.
        Formally, given a vector z, the Gumbel-softmax function s(z,tau)_i for a component i at temperature tau is
        defined as:

            s(z,tau)_i = frac{e^{(z_i + g_i) / tau}}{sum_{j=1}^M e^{(z_j + g_j) / tau}}

        where g_i is a sample from the Gumbel(0, 1) distribution. The parameter tau (temperature) controls the sharpness
        of the output distribution. As tau approaches 0, the mixing probabilities become more discrete, and as tau
        approaches infty, the mixing probabilities become more uniform. For more information we refer to

            Jang, E., Gu, Shixiang and Poole, B. "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017.

    initialize: bool
        Whether to initialize the distributional parameters with unconditional start values. Initialization can help
        to improve speed of convergence in some cases. However, it may also lead to early stopping or suboptimal
        solutions if the unconditional start values are far from the optimal values.
    """
    def __init__(self,
                 component_distribution: torch.distributions.Distribution,
                 M: int = 2,
                 hessian_mode: str = "individual",
                 tau: float = 1.0,
                 initialize: bool = False,
                 ):

        # Input Checks
        mixt_dist = get_component_distributions()
        if str(component_distribution.__class__).split(".")[-2] not in mixt_dist:
            raise ValueError(f"component_distribution must be one of the following: {mixt_dist}.")
        if not isinstance(M, int):
            raise ValueError("M must be an integer.")
        if M < 2:
            raise ValueError("M must be greater than 1.")
        if component_distribution.loss_fn != "nll":
            raise ValueError("Loss for component_distribution must be 'nll'.")
        if not isinstance(hessian_mode, str):
            raise ValueError("hessian_mode must be a string.")
        if hessian_mode not in ["individual", "grouped"]:
            raise ValueError("hessian_mode must be either 'individual' or 'grouped'.")
        if not isinstance(tau, float):
            raise ValueError("tau must be a float.")
        if tau <= 0:
            raise ValueError("tau must be greater than 0.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        # Set the parameters specific to the distribution
        param_dict = component_distribution.param_dict
        preset_gumbel_fn = partial(gumbel_softmax_fn, tau=tau)
        param_dict.update({"mix_prob": preset_gumbel_fn})
        distribution_arg_names = [f"{key}_{i}" for key in param_dict for i in range(1, M + 1)]
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=component_distribution,
                         M=M,
                         temperature=tau,
                         hessian_mode=hessian_mode,
                         univariate=True,
                         discrete=component_distribution.discrete,
                         n_dist_param=len(distribution_arg_names),
                         stabilization=component_distribution.stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=distribution_arg_names,
                         loss_fn=component_distribution.loss_fn,
                         initialize=initialize,
                         )
