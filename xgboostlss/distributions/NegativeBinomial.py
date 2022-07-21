import xgboost as xgb
import numpy as np
import pandas as pd
import math
from scipy.stats import nbinom, poisson
from scipy.special import digamma
from xgboostlss.utils import soft_plus, soft_plus_inv, stabilize_derivative

np.seterr(all="ignore")

########################################################################################################################
##################################################      NBI      #######################################################
########################################################################################################################

# When a custom objective is provided XGBoost doesn't know its response function so the user is responsible for making
# the transformation for both objective and custom evaluation metric. For objective with identity link like squared
# error this is trivial, but for other link functions like log link or inverse link the difference is significant.

# For the Python package, the behaviour of the predictions can be controlled by the output_margin parameter in the
# predict function. When using the custom_metric parameter without a custom objective, the metric function will receive
# transformed predictions since the objective is defined by XGBoost. However, when a custom objective is also provided
# along with a custom metric, then both the objective and custom metric will receive raw predictions and hence must be
# transformed using the specified response functions.

class NBI():
    """Negative Binomial Type I Distribution Class

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
        loc_fit = np.nanmean(y)
        scale_fit = np.max([((np.nanvar(y, ddof=1) - np.nanmean(y))/(np.nanmean(y)**2)), 0.1])
        location_init = NBI.param_dict_inv()["location_inv"](loc_fit)
        scale_init = NBI.param_dict_inv()["scale_inv"](scale_fit)

        start_values = np.array([location_init, scale_init])

        return start_values



    ###
    # Density Function
    ###
    @staticmethod
    def dNBI(y: np.ndarray, location: np.ndarray, scale: np.ndarray):
        """Density function.

        """
        n = 1 / scale
        p = n / (n + location)
        if len(scale) > 1:
            fy = np.where(scale > 1e-04, nbinom.logpmf(k=y, n=n, p=p), poisson.logpmf(k=y, mu=location))
        else:
            fy = poisson.logpmf(k=y, mu=location) if scale < 1e-04 else nbinom.logpmf(k=y, n=n, p=p)
        return fy


    ###
    # Quantile Function
    ###
    @staticmethod
    def qNBI(q: float, location: np.ndarray, scale: np.ndarray):
        """Quantile function.

        """
        n = 1 / scale
        p = n / (n + location)
        if len(scale) > 1:
            quant = np.where(scale > 1e-04, nbinom.ppf(q=q, n=n, p=p), poisson.ppf(q=q, mu=location))
        else:
            quant = poisson.ppf(q=q, mu=location) if scale < 1e-04 else nbinom.ppf(q=q, n=n, p=p)
        return quant


    ###
    # Random variable generation
    ###
    @staticmethod
    def rNBI(n: int, location: np.ndarray, scale: np.ndarray):
        """Random variable generation function.

        """
        n = math.ceil(n)
        p = np.random.uniform(0, 1, n)
        r = NBI.qNBI(q=p, location=location, scale=scale)
        return r



    ###
    # Location Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_location(y: np.ndarray, location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of location parameter.

        """
        grad = (y - location) / (location * (1 + location * scale))
        grad = stabilize_derivative(grad, NBI.stabilize)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_location(location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of location parameter.

        """
        hes = -1 / (location * (1 + location * scale))
        hes = stabilize_derivative(hes, NBI.stabilize)
        hes = hes * (-1) * weights
        return hes


    ###
    # Scale Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of scale parameter.

        """
        grad = -((1 / scale) ** 2) * (digamma(y + (1 / scale)) - digamma(1 / scale) - np.log(1 + location * scale) - (
                    y - location) * scale / (1 + location * scale))
        grad = stabilize_derivative(grad, NBI.stabilize)
        grad = grad * (-1) * weights
        return grad

    @staticmethod
    def hessian_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of scale parameter.

        """
        hes = -((1 / scale) ** 2) * (digamma(y + (1 / scale)) - digamma(1 / scale) - np.log(1 + location * scale) - (
                    y - location) * scale / (1 + location * scale))
        hes = -hes ** 2
        hes = np.where(hes < -1e-15, hes, -1e-15)
        hes = stabilize_derivative(hes, NBI.stabilize)
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
        preds_location = NBI.param_dict()["location"](predt[:, 0])
        preds_scale = NBI.param_dict()["scale"](predt[:, 1])


        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype=float)
        else:
            weights = data.get_weight()


        # Initialize Gradient and Hessian Matrices
        grad = np.zeros(shape=(len(target), NBI.n_dist_param()))
        hess = np.zeros(shape=(len(target), NBI.n_dist_param()))


        # Location
        grad[:, 0] = NBI.gradient_location(y=target,
                                           location=preds_location,
                                           scale=preds_scale,
                                           weights=weights)

        hess[:, 0] = NBI.hessian_location(location=preds_location,
                                          scale=preds_scale,
                                          weights=weights)

        # Scale
        grad[:, 1] = NBI.gradient_scale(y=target,
                                        location=preds_location,
                                        scale=preds_scale,
                                        weights=weights)

        hess[:, 1] = NBI.hessian_scale(y=target,
                                       location=preds_location,
                                       scale=preds_scale,
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
        preds_location = NBI.param_dict()["location"](predt[:, 0])
        preds_scale = NBI.param_dict()["scale"](predt[:, 1])

        nll = -np.nansum(NBI.dNBI(y=target, location=preds_location, scale=preds_scale))

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
            pred_dist_list.append(NBI.rNBI(n=n_samples,
                                           location=np.array([pred_params.loc[i, "location"]]),
                                           scale=np.array([pred_params.loc[i, "scale"]]))
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
            pred_quantiles_list.append(NBI.qNBI(q=quantiles[i],
                                                location=pred_params["location"],
                                                scale=pred_params["scale"]
                                                )
                                       )

        pred_quantiles = pd.DataFrame(pred_quantiles_list).T
        return pred_quantiles


