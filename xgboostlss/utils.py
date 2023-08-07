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


def exp_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used for Student-T distribution.

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

    return predt + torch.tensor(2.0, dtype=predt.dtype)


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


def softplus_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used for Student-T distribution.

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

    return predt + torch.tensor(2.0, dtype=predt.dtype)


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
    predt = torch.nan_to_num(predt, nan=float(torch.nanmean(predt))) + torch.tensor(1e-06, dtype=predt.dtype)
    predt = torch.clamp(predt, 1e-03, 1-1e-03)

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
