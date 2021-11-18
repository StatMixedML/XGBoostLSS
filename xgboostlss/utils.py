import numpy as np

###
# Response Functions
###
def identity(predt: np.ndarray):
    '''Identity mapping of predt .

    '''
    return predt


def soft_plus(predt: np.ndarray):
    '''Softplus function used to ensure predt is strictly positive.

    '''
    safe_softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    safe_softplus[safe_softplus == 0] = 1e-15
    return safe_softplus


def soft_plus1(predt: np.ndarray):
    '''Softplus function used to ensure predt is greater than 1.

    '''
    safe_softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    safe_softplus[safe_softplus <= 1] = 1 + 1e-15
    return safe_softplus
