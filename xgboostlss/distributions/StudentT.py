import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.special import polygamma
from xgboostlss.utils import *

np.seterr(all="ignore")

########################################################################################################################
###############################################      Student-T    ######################################################
########################################################################################################################

class StudentT():
    """"Student-T Distribution Class

    """

    ###
    # Specifies the number of distributional parameters
    ###
    @staticmethod
    def n_dist_param():
        """Number of distributional parameter.

        """
        n_param = 3
        return n_param


    ###
    # Parameter Dictionary
    ###
    @staticmethod
    def param_dict():
        """ Dictionary that holds the name of distributional parameter and their corresponding response functions.

        """
        param_dict = {"location": identity,
                      "scale": soft_plus,
                      "nu": soft_plus}

        return param_dict


    ###
    # Location Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_location(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray):
        """Calculates Gradient of location parameter.

        """
        s2 = scale ** 2
        dsq = ((y - location) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        grad = (omega * (y - location)) / s2
        return (grad * (-1))

    @staticmethod
    def hessian_location(scale: np.ndarray, nu: np.ndarray):
        """Calculates Hessian of location parameter.

        """
        hes = -(nu + 1) / ((nu + 3) * (scale ** 2))
        return (hes * (-1))



    ###
    # Scale Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray):
        """Calculates Gradient of scale parameter.

        """
        s2 = scale ** 2
        dsq = ((y - location) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        grad = (omega * dsq - 1) / scale
        return (grad * (-1))

    @staticmethod
    def hessian_scale(scale: np.ndarray, nu: np.ndarray):
        """Calculates Hessian of scale parameter.

        """
        s2 = scale ** 2
        hes = -(2 * nu) / ((nu + 3) * s2)
        return (hes * (-1))



    ###
    # Nu Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_nu(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray):
        """Calculates Gradient of nu parameter.

        """
        s2 = scale ** 2
        dsq = ((y - location) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        dsq3 = 1 + (dsq / nu)
        v2 = nu / 2
        v3 = (nu + 1) / 2
        grad = -np.log(dsq3) + (omega * dsq - 1) / nu + polygamma(0, v3) - polygamma(0, v2)
        grad = grad / 2
        return (grad * (-1))

    @staticmethod
    def hessian_nu(y: np.ndarray, location: np.ndarray, scale: np.ndarray, nu: np.ndarray):
        """Calculates Hessian of nu parameter.

        """
        v2 = nu / 2
        v3 = (nu + 1) / 2
        hes = polygamma(1, v3) - polygamma(1, v2) + (2 * (nu + 5)) / (nu * (nu + 1) * (nu + 3))
        hes = hes / 4
        hes = np.where(hes < -1e-15, hes, -1e-15)
        return (hes * (-1))



    ###
    # Custom Objective Function
    ###
    def Dist_Objective(predt: np.ndarray, data: xgb.DMatrix):
        """A customized objective function to train each distributional parameter using custom gradient and hessian.

        """

        target = data.get_label()

        # When num_class!= 0, preds has shape (n_obs, n_classes)
        preds_location = StudentT.param_dict()["location"](predt[:, 0])
        preds_scale = StudentT.param_dict()["scale"](predt[:, 1])
        preds_nu = StudentT.param_dict()["nu"](predt[:, 2])

        # Initialize Gradient and Hessian Matrices
        grad = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)
        hess = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)

        # Location
        grad[:, 0] = StudentT.gradient_location(y=target, location=preds_location, scale=preds_scale, nu=preds_nu)
        hess[:, 0] = StudentT.hessian_location(scale=preds_scale, nu=preds_nu)

        # Scale
        grad[:, 1] = StudentT.gradient_scale(y=target, location=preds_location, scale=preds_scale, nu=preds_nu)
        hess[:, 1] = StudentT.hessian_scale(scale=preds_scale, nu=preds_nu)

        # Nu
        grad[:, 2] = StudentT.gradient_nu(y=target, location=preds_location, scale=preds_scale, nu=preds_nu)
        hess[:, 2] = StudentT.hessian_nu(y=target, location=preds_location, scale=preds_scale, nu=preds_nu)

        # Reshaping
        grad = grad.reshape((predt.shape[0] * predt.shape[1], 1))
        hess = hess.reshape((predt.shape[0] * predt.shape[1], 1))

        return grad, hess


    ###
    # Custom Evaluation Metric
    ###
    def Dist_Metric(predt: np.ndarray, data: xgb.DMatrix):
        """A customized evaluation metric that evaluates the predictions using the
        negative log-likelihood

        """
        target = data.get_label()
        preds_location = StudentT.param_dict()["location"](predt[:, 0])
        preds_scale = StudentT.param_dict()["scale"](predt[:, 1])
        preds_nu = StudentT.param_dict()["nu"](predt[:, 2])

        nll = -np.sum(student_t.logpdf(x=target, loc=preds_location, scale=preds_scale, df=preds_nu))
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
            pred_dist_list.append(student_t.rvs(loc=pred_params.loc[i, "location"],
                                                scale=pred_params.loc[i, "scale"],
                                                df=pred_params.loc[i, "nu"],
                                                size=n_samples,
                                                random_state=seed)
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
            pred_quantiles_list.append(student_t.ppf(quantiles[i],
                                                     loc=pred_params["location"],
                                                     scale=pred_params["scale"],
                                                     df=pred_params["nu"])
                                       )

        pred_quantiles = pd.DataFrame(pred_quantiles_list).T
        return pred_quantiles


