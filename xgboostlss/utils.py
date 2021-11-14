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
    return np.log(1 + np.exp(predt))
