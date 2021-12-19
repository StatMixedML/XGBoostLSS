import numpy as np
import torch
from torch.autograd import grad

###
# Response Functions
###
def identity(predt: np.ndarray):
    """Identity mapping of predt .

    """
    return predt


def soft_plus(predt: np.ndarray):
    """Softplus function used to ensure predt is strictly positive.

    """
    predt = np.log1p(np.exp(-np.abs(predt))) + np.maximum(predt, 0)
    predt[predt == 0] = 1e-15
    return predt


def soft_plus_inv(predt: np.ndarray):
    """Inverse of softplus function.

    """
    predt = np.log(np.exp(predt) - 1)
    return predt



###
# Stabilization of Gradient and Hessian
###
def stabilize_derivative(input_der: np.ndarray,  type: str = "None"):
    """Function that stabilizes Gradients and Hessians.

    As XGBoostLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important
    that these are comparable in magnitude for all distributional parameters. Due to imbalances regarding the ranges,
    the algorithm might become unstable so that it does not converge (or converge very slowly) to the optimal solution.

    Another way to improve convergence might be to standardize the response variable. This is especially useful if the
    range of the response differs strongly from the range of the Gradients and Hessians. Both, the stabilization and
    the standardization of the response are not always advised but need to be carefully considered given the data at hand.

    Source: https://github.com/boost-R/gamboostLSS/blob/7792951d2984f289ed7e530befa42a2a4cb04d1d/R/helpers.R#L173



    Parameters
    ----------
    input_der : np.ndarray
        Either Gradient or Hessian.
    type: str
        Stabilization method. Can be either "None", "MAD" or "L2".

    Returns
    -------
    stab_der : np.ndarray
        Stabilized Gradient or Hessian.

    """

    if type == "MAD":
        div = np.nanmedian(np.abs(input_der - np.nanmedian(input_der)))
        div = np.where(div < 1e-04, 1e-04, div)
        stab_der = input_der/div

    if type == "L2":
        div = np.sqrt(np.nanmean(input_der**2))
        div = np.where(div < 1e-04, 1e-04, div)
        div = np.where(div > 10000, 10000, div)
        stab_der = input_der/div

    if type == "None":
        stab_der = input_der

    return stab_der



def auto_grad(metric: torch.Tensor, parameter: torch.Tensor, n: int):
    """Function for automatic differentiation and calculation of derivatives using PyTorch AutoGrad.

    Parameters
    ----------
    metric: Tensor
        Input, usually sum of negative log-likelihood.
    parameter: Tensor
        Distributional parameter for which to calculate derivative.
    n: int
        Order of derivative: 1=Gradient, 2=Hessian.

    Returns
    -------
    deriv : Tensor
        Tensor of derivatives.

    """

    for i in range(n):
        deriv = grad(metric, parameter, create_graph=True)[0]
        metric = deriv.nansum()
    deriv_np = np.round(deriv.detach().numpy(), 5)

    return deriv_np