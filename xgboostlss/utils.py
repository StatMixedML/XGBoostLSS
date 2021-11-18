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
    predt = np.log1p(np.exp(-np.abs(predt))) + np.maximum(predt, 0)
    predt[predt == 0] = 1e-15
    return predt


def soft_plus1(predt: np.ndarray):
    '''Softplus function used to ensure predt is greater than 1.

    '''
    predt = np.log1p(np.exp(-np.abs(predt))) + np.maximum(predt, 0)
    predt[predt <= 1] = 1 + 1e-15
    return predt
