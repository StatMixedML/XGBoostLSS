import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboostlss.utils import *



########################################################################################################################
###############################################      Gaussian      #####################################################
########################################################################################################################
class Gaussian():
    """"Abstract Distribution Class

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
        """ Dictionary that holds the name of distributional parameter and their correspondig repsonse functions.

        """
        param_dict = {"location": identity,
                      "scale": soft_plus}

        return param_dict


    ###
    # Location Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_location(y: np.ndarray, location: np.ndarray, scale: np.ndarray):
        """Calculates Gradient of location parameter.

        """
        grad = (1/(scale**2)) * (y - location)
        return(grad*(-1))


    @staticmethod
    def hessian_location(scale: np.ndarray):
        """Calculates Hessian of location parameter.

        """
        hes = -(1/(scale**2))
        return(hes*(-1))


    ###
    # Scale Parameter gradient and hessian
    ###
    @staticmethod
    def gradient_scale(y: np.ndarray, location: np.ndarray, scale: np.ndarray):
        """Calculates Gradient of scale parameter.

        """
        grad = ((y - location)**2 - scale**2)/(scale**3)
        return(grad*(-1))

    @staticmethod
    def hessian_scale(scale: np.ndarray):
        '''Calculates Hessian of scale parameter.

        '''
        hes = -(2/(scale**2))
        return(hes*(-1))



    ###
    # Custom Objective Function
    ###
    def Dist_Objective(predt: np.ndarray, data: xgb.DMatrix):
        """A customized objective function to train each distributional parameter using custom gradient and hessian.

        """

        target = data.get_label()

        # When num_class!= 0, preds has shape (n_obs, n_classes)
        preds_location = predt[:, 0]
        preds_scale = soft_plus(predt[:, 1])

        # Initialize Gradient and Hessian Matrices
        grad = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)
        hess = np.zeros((predt.shape[0], predt.shape[1]), dtype=float)

        # Location
        grad[:, 0] = Gaussian.gradient_location(y=target, location=preds_location, scale=preds_scale)
        hess[:, 0] = Gaussian.hessian_location(scale=preds_scale)

        # Scale
        grad[:, 1] = Gaussian.gradient_scale(y=target, location=preds_location, scale=preds_scale)
        hess[:, 1] = Gaussian.hessian_scale(scale=preds_scale)

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
        preds_location = predt[:, 0]
        preds_scale = soft_plus(predt[:, 1])

        nll = -np.sum(norm.logpdf(x=target, loc=preds_location, scale=preds_scale))

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
            pred_dist_list.append(norm.rvs(loc=pred_params.loc[i,"location"],
                                           scale=pred_params.loc[i,"scale"],
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
            pred_quantiles_list.append(norm.ppf(quantiles[i],
                                                loc = pred_params["location"],
                                                scale = pred_params["scale"])
                                       )

        pred_quantiles = pd.DataFrame(pred_quantiles_list).T
        return pred_quantiles


