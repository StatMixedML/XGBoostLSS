import xgboost as xgb
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from collections import ChainMap
from collections import OrderedDict
from xgboostlss.utils import *

np.seterr(all="ignore")

########################################################################################################################
###############################################      Expectile      #####################################################
########################################################################################################################

# When a custom objective is provided XGBoost doesn't know its response function so the user is responsible for making
# the transformation for both objective and custom evaluation metric. For objective with identity link like squared
# error this is trivial, but for other link functions like log link or inverse link the difference is significant.

# For the Python package, the behaviour of the predictions can be controlled by the output_margin parameter in the
# predict function. When using the custom_metric parameter without a custom objective, the metric function will receive
# transformed predictions since the objective is defined by XGBoost. However, when a custom objective is also provided
# along with a custom metric, then both the objective and custom metric will receive raw predictions and hence must be
# transformed using the specified response functions.

class Expectile():
    """Expectile Distribution Class

    """

    # Specifies the number of distributional parameters
    @staticmethod
    def n_dist_param():
        """Number of distributional parameter.

        """
        n_param = len(Expectile.expectiles)
        return n_param

    ###
    # Parameter Dictionary
    ###
    @staticmethod
    def param_dict():
        """ Dictionary that holds the expectiles and their corresponding response functions.

        """
        param_dict = []
        for i in range(len(Expectile.expectiles)):
            param_dict.append({"expectile_" + str(Expectile.expectiles[i]): identity})

        param_dict = dict(ChainMap(*param_dict))
        param_dict = OrderedDict(sorted(param_dict.items(), key=lambda x: x[0]))

        return param_dict


    ###
    # Starting Values
    ###
    @staticmethod
    def initialize(y: np.ndarray):
        """ Function that calculates the starting values, for each distributional parameter individually.

        y: np.ndarray
            Data from which starting values are calculated.

        """

        expect_init=[]
        for i in range(len(Expectile.expectiles)):
            expect_init.append(np.mean(y))

        start_values = np.array([expect_init])
        return start_values


    ###
    # Expectile Normal Distribution Function
    ###
    @staticmethod
    def expectile_pnorm(tau, m=0, sd=1):
        """Normal Expectile Distribution Function. For more details and distributions see https://rdrr.io/cran/expectreg/man/enorm.html

        Parameters
        ----------
        tau : np.ndarray
            Vector of expectiles from the respective distribution.
        m : np.ndarray
            Mean of the Normal distribution.
        sd : np.ndarray
            Standard Deviation of the Normal distribution.

        Returns
        -------
        tau : np.ndarray
            Vector of expectiles for the Normal distribution.

        """

        z = (tau - m) / sd
        p = norm.cdf(z, loc=m, scale=sd)
        d = norm.pdf(z, loc=m, scale=sd)
        u = -d - z * p
        tau = u / (2 * u + z)
        return tau

    ###
    # Numeric Expectiles
    ###
    @staticmethod
    def expect_norm(tau, m=0, sd=1):
        """Calculate expectiles from Normal distribution for given tau values. For more details and distributions see https://rdrr.io/cran/expectreg/man/enorm.html

        Parameters
        ----------
        tau : np.ndarray
            Vector of taummetries with values between 0 and 1.
        m : np.ndarray
            Mean of the Normal distribution.
        sd : np.ndarray
            Standard Deviation of the Normal distribution.

        """

        tau[tau > 1 or tau < 0] = np.nan
        zz = 0 * tau
        lower = np.array(-10, dtype="float")
        lower = np.repeat(lower[np.newaxis, ...], len(tau), axis=0)
        upper = np.array(10, dtype="float")
        upper = np.repeat(upper[np.newaxis, ...], len(tau), axis=0)
        diff = 1
        index = 0
        while (diff > 1e-10) and (index < 1000):
            root = Expectile.expectile_pnorm(zz) - tau
            root[math.isnan(root)] = 0
            lower[root < 0] = zz[root < 0]
            upper[root > 0] = zz[root > 0]
            zz = (upper + lower) / 2
            diff = np.nanmax(np.abs(root))
            index = index + 1

        zz[math.isnan(tau)] = np.nan
        return zz * sd + m



    ###
    # Gradient and Hessian
    ###
    @staticmethod
    def gradient_expectile(y: np.ndarray, expectile: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of expectile.

        """
        grad = 2 * tau * (y - expectile) * ((y - expectile) > 0) - 2 * (1 - tau) * (expectile - y) * ((y - expectile) < 0) + 0 * ((y - expectile) == 0)
        grad = stabilize_derivative(grad, Expectile.stabilize)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_expectile(y: np.ndarray, expectile: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of expectile.

        """
        hes = -2 * tau * ((y - expectile) > 0) - 2 * (1 - tau) * ((y - expectile) < 0) + 0 * ((y - expectile) == 0)
        hes = stabilize_derivative(hes, Expectile.stabilize)
        hes = hes * (-1) * weights
        return hes


    ###
    # Expectile Loss Function
    ###
    @staticmethod
    def expectile_loss(y: np.ndarray, expectile: np.ndarray, tau: np.ndarray):
        loss = tau * (y - expectile) ** 2 * ((y - expectile) >= 0) + (1 - tau) * (y - expectile) ** 2 * ((y - expectile) < 0)
        return loss.sum()




    ###
    # Custom Objective Function
    ###
    def Dist_Objective(predt: np.ndarray, data: xgb.DMatrix):
        """A customized objective function to train each distributional parameter using custom gradient and hessian.

        """

        target = data.get_label()

        # When num_class!= 0, preds has shape (n_obs, n_dist_param).
        # Each element in a row represents a raw prediction (leaf weight, hasn't gone through response function yet).
        preds_expectile = predt

        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype=float)
        else:
            weights = data.get_weight()


        # Initialize Gradient and Hessian Matrices
        grad = np.zeros(shape=(len(target), len(Expectile.expectiles)))
        hess = np.zeros(shape=(len(target), len(Expectile.expectiles)))

        for i in range(len(Expectile.expectiles)):
            grad[:, i] = Expectile.gradient_expectile(y=target,
                                                      expectile=preds_expectile[:, i],
                                                      tau=Expectile.expectiles[i],
                                                      weights=weights
                                                      )

            hess[:, i] = Expectile.hessian_expectile(y=target,
                                                     expectile=preds_expectile[:, i],
                                                     tau=Expectile.expectiles[i],
                                                     weights=weights
                                                     )

        # Reshaping
        grad = grad.flatten()
        hess = hess.flatten()

        return grad, hess



    ###
    # Custom Evaluation Metric
    ###
    def Dist_Metric(predt: np.ndarray, data: xgb.DMatrix):
        """A customized evaluation metric that evaluates the predictions using the negative log-likelihood.

        """
        target = data.get_label()

        # Using a custom objective function, the custom metric receives raw predictions which need to be transformed
        # with the corresponding response function.
        preds_expectile = predt

        loss_expectile=[]
        for i in range(len(Expectile.expectiles)):
            loss_expectile.append(
                Expectile.expectile_loss(y=target,
                                         expectile=preds_expectile[:, i],
                                         tau=Expectile.expectiles[i]
                                         )
            )


        nll = np.nanmean(loss_expectile)

        return "NegLogLikelihood", nll


