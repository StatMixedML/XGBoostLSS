import torch


def identity_fn(predt: torch.tensor) -> torch.tensor:
    """
    Identity mapping of predt.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    return predt


def exp_fn(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.exp(predt)
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def log_fn(predt: torch.tensor) -> torch.tensor:
    """
    Inverse of exp_fn function.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.log(predt)
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + float(1e-06)

    return predt


def softplus_fn(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.log1p(torch.exp(-torch.abs(predt))) + torch.maximum(predt, torch.tensor(0.))
    predt[predt == 0] = torch.tensor(1e-06, dtype=predt.dtype)
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def softplusinv_fn(predt: torch.tensor) -> torch.tensor:
    """
    Inverse of softplus_fn function.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = predt + torch.log(-torch.expm1(-predt))
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def sigmoid_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to (0,1).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.sigmoid(predt)
    predt = torch.clamp(predt, 1e-06, 1-1e-06)
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt

def sigmoidinv_fn(predt: torch.tensor) -> torch.tensor:
    """
    Inverse of sigmoid_fn function.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.log(predt / (1 - predt))
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-6, dtype=predt.dtype)

    return predt

def relu_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to max(0, predt).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.relu(predt)
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-6, dtype=predt.dtype)

    return predt

def reluinv_fn(predt: torch.tensor) -> torch.tensor:
    """
    Inverse of relu_fn function. Since ReLU sets all negative values to zero, it loses information about the sign of
    the input. Therefore, it is not possible to uniquely recover the original input from the output of the ReLU
    function. As a result, the ReLU function does not have a direct inverse. Hence, we use the identity function as
    the inverse of the ReLU function.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """

    return predt
