import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import t, norm
import math
from scipy.special import gammaln, digamma, polygamma
from xgboostlss.utils import *

np.seterr(all="ignore") 

########################################################################################################################
#################################################     BCT    ##########################################################
########################################################################################################################

# When a custom objective is provided XGBoost doesn't know its response function so the user is responsible for making
# the transformation for both objective and custom evaluation metric. For objective with identity link like squared
# error this is trivial, but for other link functions like log link or inverse link the difference is significant.

# For the Python package, the behaviour of the predictions can be controlled by the output_margin parameter in the
# predict function. When using the custom_metric parameter without a custom objective, the metric function will receive
# transformed predictions since the objective is defined by XGBoost. However, when a custom objective is also provided
# along with a custom metric, then both the objective and custom metric will receive raw predictions and hence must be
# transformed using the specified response functions.



class BCT():
    """"Box-Cox t (BCT) Distribution Class

    """

    ###
    # Specifies the number of distributional parameters
    ###
    @staticmethod
    def n_dist_param():
        """Number of distributional parameter.

        """
        n_param = 4
        return n_param


    ###
    # Parameter Dictionary
    ###
    @staticmethod
    def param_dict():
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.

        """
        param_dict = {"location": soft_plus,
                      "scale": soft_plus,
                      "nu": identity,
                      "tau": soft_plus
                      }

        return param_dict


    ###
    # Density Function
    ###
    @staticmethod
    def dBCCG(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, log=False):
        """Helper density function.

        """
        if len(nu) > 1:
            z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        elif nu != 0:
            z = (((y / location) ** nu - 1) / (nu * scale))
        else:
            z = np.log(y / location) / scale
        loglik = nu * np.log(y / location) - np.log(scale) - (z ** 2) / 2 - np.log(y) - (np.log(2 * np.pi)) / 2
        loglik = loglik - np.log(norm.cdf(1 / (scale * np.abs(nu))))
        if log == False:
            ft = np.exp(loglik)
        else:
            ft = loglik
        return ft


    @staticmethod
    def dBCT(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, log=False):
        """Density function.

        """
        if len(nu) > 1:
            z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        elif nu != 0:
            z = (((y / location) ** nu - 1) / (nu * scale))
        else:
            z = np.log(y / location) / scale
        loglik = (nu - 1) * np.log(y) - nu * np.log(location) - np.log(scale)
        fTz = gammaln((tau + 1) / 2) - gammaln(tau / 2) - 0.5 * np.log(tau) - gammaln(0.5)
        fTz = fTz - ((tau + 1) / 2) * np.log(1 + (z * z) / tau)
        loglik = loglik + fTz - np.log(t.cdf(1 / (scale * np.abs(nu)), df=tau))
        if len(tau) > 1:
            loglik = np.where(tau > 1e+06, BCT.dBCCG(y, location, scale, nu, log=True), loglik)
        elif tau > 1e+06:
            loglik = BCT.dBCCG(y, location, scale, nu, log=True)
        ft = np.exp(loglik) if log == False else loglik
        return ft



    ###
    # Quantile Function
    ###
    @staticmethod
    def qBCT(p: float, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, lower_tail=True,
             log_p=False):
        """Quantile function.

        """
        if log_p == True:
            p = np.exp(p)
        else:
            p = p
        if lower_tail == True:
            p = p
        else:
            p = 1 - p
        if len(nu) > 1:
            z = np.where((nu <= 0), t.ppf(p * t.cdf(1 / (scale * np.abs(nu)), tau), tau),
                         t.ppf(1 - (1 - p) * t.cdf(1 / (scale * abs(nu)), tau), tau))
        else:
            z = t.ppf(p * t.cdf(1 / (scale * np.abs(nu)), tau), tau) if nu <= 0 else t.ppf(
                1 - (1 - p) * t.cdf(1 / (scale * abs(nu)), tau), tau)
        if len(nu) > 1:
            ya = np.where(nu != 0, location * (nu * scale * z + 1) ** (1 / nu), location * np.exp(scale * z))
        elif nu != 0:
            ya = location * (nu * scale * z + 1) ** (1 / nu)
        else:
            ya = location * np.exp(scale * z)
        return ya


    ###
    # Random variable generation
    ###
    def rBCT(n: int, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray):
        """Random variable generation function.

        """
        n = math.ceil(n)
        p = np.random.uniform(0, 1, n)
        r = BCT.qBCT(p, location=location, scale=scale, nu=nu, tau=tau)
        return r



    ###
    # Location Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_location(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of location parameter.

        """
        z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        w = (tau + 1) / (tau + z ** 2)
        grad = (w * z) / (location * scale) + (nu / location) * (w * (z ** 2) - 1)
        grad = stabilize_derivative(grad)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_location(location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of location parameter.

        """
        hes = -(tau + 2 * nu * nu * scale * scale * tau + 1) / (tau + 3)
        hes = hes / (location * location * scale * scale)
        hes = stabilize_derivative(hes)
        hes = hes * (-1) * weights
        return hes



    ###
    # Scale Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of scale parameter.

        """
        z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        w = (tau + 1) / (tau + z ** 2)
        h = t.pdf(1 / (scale * np.abs(nu)), df=tau) / t.cdf(1 / (scale * np.abs(nu)), df=tau)
        grad = (w * (z ** 2) - 1) / scale + h / (scale ** 2 * np.abs(nu))
        grad = stabilize_derivative(grad)
        grad = grad * (-1) * weights
        return grad



    @staticmethod
    def hessian_scale(scale: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of scale parameter.

        """
        hes = -2 * tau / (scale ** 2 * (tau + 3))
        hes = stabilize_derivative(hes)
        hes = hes * (-1) * weights
        return hes



    ###
    # Nu Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_nu(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of nu parameter.

        """
        z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        w = (tau + 1) / (tau + z ** 2)
        h = t.pdf(1 / (scale * np.abs(nu)), df=tau) / t.cdf(1 / (scale * np.abs(nu)), df=tau)
        grad = ((w * z ** 2) / nu) - np.log(y / location) * (w * z ** 2 + ((w * z) / (scale * nu)) - 1)
        grad = grad + np.sign(nu) * h / (scale * nu ** 2)
        grad = stabilize_derivative(grad)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_nu(scale: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of nu parameter.

        """
        hes = -7 * (scale ** 2) / 4
        hes = stabilize_derivative(hes)
        hes = hes * (-1) * weights
        return hes


    ###
    # Tau Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_tau(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray, tau: np.ndarray, weights: np.ndarray):
        """Calculates Gradient of tau parameter.

        """
        z = np.where(nu != 0, (((y / location) ** nu - 1) / (nu * scale)), np.log(y / location) / scale)
        w = (tau + 1) / (tau + z ** 2)
        j = (np.log(t.cdf(1 / (scale * np.abs(nu)), df=tau + 0.01)) - np.log(
            t.cdf(1 / (scale * abs(nu)), df=tau))) / 0.01
        grad = -0.5 * np.log(1 + (z ** 2) / tau) + (w * (z ** 2)) / (2 * tau)
        grad = grad + 0.5 * digamma((tau + 1) / 2) - 0.5 * digamma(tau / 2) - 1 / (2 * tau) - j
        grad = stabilize_derivative(grad)
        grad = grad * (-1) * weights
        return grad


    @staticmethod
    def hessian_tau(tau: np.ndarray, weights: np.ndarray):
        """Calculates Hessian of tau parameter.

        """
        hes = polygamma(1, ((tau + 1) / 2)) - polygamma(1, tau / 2) + 2 * (tau + 5) / (tau * (tau + 1) * (tau + 3))
        hes = hes / 4
        hes = np.where(hes < -1e-15, hes, -1e-15)
        hes = stabilize_derivative(hes)
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
        preds_location = BCT.param_dict()["location"](predt[:, 0])
        preds_scale = BCT.param_dict()["scale"](predt[:, 1])
        preds_nu = BCT.param_dict()["nu"](predt[:, 2])
        preds_tau = BCT.param_dict()["tau"](predt[:, 3])


        # Weights
        if data.get_weight().size == 0:
            # Use 1 as weight if no weights are specified
            weights = np.ones_like(target, dtype=float)
        else:
            weights = data.get_weight()


        # Initialize Gradient and Hessian Matrices
        grad = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)
        hess = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)

        # Location
        grad[:, 0] = BCT.gradient_location(y=target,
                                           location=preds_location,
                                           scale=preds_scale,
                                           nu=preds_nu,
                                           tau=preds_tau,
                                           weights=weights)

        hess[:, 0] = BCT.hessian_location(location=preds_location,
                                          scale=preds_scale,
                                          nu=preds_nu,
                                          tau=preds_tau,
                                          weights=weights)

        # Scale
        grad[:, 1] = BCT.gradient_scale(y=target,
                                        location=preds_location,
                                        scale=preds_scale,
                                        nu=preds_nu,
                                        tau=preds_tau,
                                        weights=weights)

        hess[:, 1] = BCT.hessian_scale(scale=preds_scale,
                                       tau=preds_tau,
                                       weights=weights)

        # Nu
        grad[:, 2] = BCT.gradient_nu(y=target,
                                     location=preds_location,
                                     scale=preds_scale,
                                     nu=preds_nu,
                                     tau=preds_tau,
                                     weights=weights)

        hess[:, 2] = BCT.hessian_nu(scale=preds_scale,
                                    weights=weights)

        # Tau
        grad[:, 3] = BCT.gradient_tau(y=target,
                                      location=preds_location,
                                      scale=preds_scale,
                                      nu=preds_nu,
                                      tau=preds_tau,
                                      weights=weights)

        hess[:, 3] = BCT.hessian_tau(tau=preds_tau,
                                     weights=weights)

        # Reshaping
        grad = grad.reshape((predt.shape[0] * predt.shape[1], 1))
        hess = hess.reshape((predt.shape[0] * predt.shape[1], 1))

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
        preds_location = BCT.param_dict()["location"](predt[:, 0])
        preds_scale = BCT.param_dict()["scale"](predt[:, 1])
        preds_nu = BCT.param_dict()["nu"](predt[:, 2])
        preds_tau = BCT.param_dict()["tau"](predt[:, 3])

        nll = -np.sum(BCT.dBCT(y=target,
                               location=preds_location,
                               scale=preds_scale,
                               nu=preds_nu,
                               tau=preds_tau,
                               log=True))
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
            pred_dist_list.append(BCT.rBCT(n=n_samples,
                                           location=np.array([pred_params.loc[i, "location"]]),
                                           scale=np.array([pred_params.loc[i, "scale"]]),
                                           nu=np.array([pred_params.loc[i, "nu"]]),
                                           tau=np.array([pred_params.loc[i, "tau"]])
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
            pred_quantiles_list.append(BCT.qBCT(p=quantiles[i],
                                                location=pred_params["location"],
                                                scale=pred_params["scale"],
                                                nu=pred_params["nu"],
                                                tau=pred_params["tau"])
                                       )

        pred_quantiles = pd.DataFrame(pred_quantiles_list).T
        return pred_quantiles

