import xgboost as xgb
import numpy as np
import pandas as pd
import math
from scipy.stats import gamma
from scipy.special import polygamma, loggamma
from lss_xgboost.utils import *

np.seterr(all="ignore")

########################################################################################################################
#################################################      Gamma      ######################################################
########################################################################################################################

# When a custom objective is provided XGBoost doesn't know its response function so the user is responsible for making
# the transformation for both objective and custom evaluation metric. For objective with identity link like squared
# error this is trivial, but for other link functions like log link or inverse link the difference is significant.

# For the Python package, the behaviour of the predictions can be controlled by the output_margin parameter in the
# predict function. When using the custom_metric parameter without a custom objective, the metric function will receive
# transformed predictions since the objective is defined by XGBoost. However, when a custom objective is also provided
# along with a custom metric, then both the objective and custom metric will receive raw predictions and hence must be
# transformed using the specified response functions.

class Gamma():
    """Gamma Distribution Class

    """

    # Specifies the number of distributional parameters
    @staticmethod
    def n_dist_param():
        """Number of distributional parameter.

        """
        n_param = 2
        return n_param


    ###
    # Parameter Dictionary
    ###
    @staticmethod
    def param_dict():
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.

        """
        param_dict = {"location": soft_plus,
                      "scale": soft_plus}

        return param_dict

    ###
    # Inverse Parameter Dictionary
    ###
    @staticmethod
    def param_dict_inv():
        """ Dictionary that holds the name of distributional parameter and their corresponding link functions.

        """
        param_dict_inv = {"location_inv": soft_plus_inv,
                          "scale_inv": soft_plus_inv}

        return param_dict_inv


    ###
    # Starting Values
    ###
    @staticmethod
    def initialize(y: np.ndarray):
        """ Function that calculates the starting values, for each distributional parameter individually.

        y: np.ndarray
            Data from which starting values are calculated.

        """
        loc_fit, scale_fit = np.nanmean(y), np.nanstd(y)
        location_init = Gamma.param_dict_inv()["location_inv"](loc_fit)
        scale_init = Gamma.param_dict_inv()["scale_inv"](scale_fit)

        start_values = np.array([location_init, scale_init])

        return start_values

    ###
    # Density Function
    ###
    @staticmethod
    def dGamma(y: np.ndarray, location: np.ndarray, scale: np.ndarray, log=True):
        """Density function.

        """
        loglik = (1 / scale ** 2) * np.log(y / (location * scale ** 2)) - y / (location * scale ** 2) - np.log(
            y) - loggamma(1 / scale ** 2)
        loglik = np.exp(loglik) if log == False else loglik
        return loglik


    ###
    # Quantile Function
    ###
    @staticmethod
    def qGamma(p: float, location: np.ndarray, scale: np.ndarray):
        """Quantile function.

        """
        q = gamma.ppf(p, a=1 / scale ** 2, scale=location * scale ** 2)
        return q


    ###
    # Random variable generation
    ###
    def rGamma(n: int, location: np.ndarray, scale: np.ndarray):
        """Random variable generation function.

        """
        n = math.ceil(n)
        p = np.random.uniform(0, 1, n)
        r = Gamma.qGamma(p, location=location, scale=scale)
        return r



    ###
    # Location Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_location(y: np.ndarray, location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of location parameter.

        """
        grad = (y - location) / ((scale ** 2) * (location ** 2))
        grad = stabilize_derivative(grad, Gamma.stabilize)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_location(location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of location parameter.

        """
        hes = -1 / ((scale ** 2) * (location ** 2))
        hes = stabilize_derivative(hes, Gamma.stabilize)
        hes = hes * (-1) * weights
        return hes


    ###
    # Scale Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of scale parameter.

        """
        grad = (2 / scale ** 3) * ((y / location) - np.log(y) + np.log(location) + np.log(scale ** 2) - 1 + polygamma(0,(1 / (scale ** 2))))
        grad = stabilize_derivative(grad, Gamma.stabilize)
        grad = grad * (-1) * weights
        return grad

    @staticmethod
    def hessian_scale(scale: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of scale parameter.

        """
        hes = (4/scale**4) - (4/scale**6) * polygamma(1, (1/scale**2))
        hes = stabilize_derivative(hes, Gamma.stabilize)
        hes = hes * (-1) * weights
        return hes



    ###
    # Custom Objective Function
    ###
    def Dist_Objective(predt: np.ndarray, data: xgb.DMatrix):
        """A customized objective function to train each distributional parameter using custom gradient and hessian.

        """

        target = data.get_label()

        # When num_class!= 0, preds has shape (n_obs, n_dist_param).
        # Each element in a row represents a raw prediction (leaf weight, hasn't gone through response function yet).
        preds_location = Gamma.param_dict()["location"](predt[:, 0])
        preds_scale = Gamma.param_dict()["scale"](predt[:, 1])


        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype=float)
        else:
            weights = data.get_weight()


        # Initialize Gradient and Hessian Matrices
        grad = np.zeros(shape=(len(target), Gamma.n_dist_param()))
        hess = np.zeros(shape=(len(target), Gamma.n_dist_param()))


        # Location
        grad[:, 0] = Gamma.gradient_location(y=target,
                                             location=preds_location,
                                             scale=preds_scale,
                                             weights=weights)

        hess[:, 0] = Gamma.hessian_location(location=preds_location,
                                            scale=preds_scale,
                                            weights=weights)

        # Scale
        grad[:, 1] = Gamma.gradient_scale(y=target,
                                          location=preds_location,
                                          scale=preds_scale,
                                          weights=weights)

        hess[:, 1] = Gamma.hessian_scale(scale=preds_scale,
                                         weights=weights)

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
        preds_location = Gamma.param_dict()["location"](predt[:, 0])
        preds_scale = Gamma.param_dict()["scale"](predt[:, 1])

        nll = -np.nansum(Gamma.dGamma(y=target,
                                      location=preds_location,
                                      scale=preds_scale,
                                      log=True)
                         )

        return "NegLogLikelihood", nll


    ###
    # Function for drawing random samples from predicted distribution
    ###
    def pred_dist_rvs(pred_params: pd.DataFrame, n_samples: int, seed: int):
        """
        Function that draws n_samples from a predicted response distribution.

        pred_params: pd.DataFrame
            Dataframe with predicted distributional parameters.
        n_samples: int
            Number of sample to draw from predicted response distribution.
        seed: int
            Manual seed.
        Returns
        -------
        pd.DataFrame with n_samples drawn from predicted response distribution.

        """
        pred_dist_list = []

        for i in range(pred_params.shape[0]):
            pred_dist_list.append(Gamma.rGamma(n=n_samples,
                                               location=np.array([pred_params.loc[i, "location"]]),
                                               scale=np.array([pred_params.loc[i, "scale"]])
                                               )
                                  )

        pred_dist = pd.DataFrame(pred_dist_list)
        return pred_dist

    ###
    # Function for calculating quantiles from predicted distribution
    ###
    def pred_dist_quantile(quantiles: list, pred_params: pd.DataFrame):
        """
        Function that calculates the quantiles from the predicted response distribution.

        quantiles: list
            Which quantiles to calculate
        pred_params: pd.DataFrame
            Dataframe with predicted distributional parameters.

        Returns
        -------
        pd.DataFrame with calculated quantiles.

        """
        pred_quantiles_list = []

        for i in range(len(quantiles)):
            pred_quantiles_list.append(Gamma.qGamma(p=quantiles[i],
                                                    location=pred_params["location"],
                                                    scale=pred_params["scale"])
                                       )

        pred_quantiles = pd.DataFrame(pred_quantiles_list).T
        return pred_quantiles
