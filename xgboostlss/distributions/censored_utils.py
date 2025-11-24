import numpy as np
import torch
import xgboost as xgb
from typing import List, Tuple
from .distribution_utils import DistributionClass


class CensoredMixin(DistributionClass):
    """
    Mixin to add interval-censoring support to a distribution.
    Overrides objective_fn and metric_fn to dispatch to censored loss.
    """
    def objective_fn(self, predt: np.ndarray, data: xgb.DMatrix):
        lower = data.get_float_info("label_lower_bound")
        upper = data.get_float_info("label_upper_bound")
        if lower.size == 0 and upper.size == 0:
            return super().objective_fn(predt, data)
        if data.get_weight().size == 0:
            # initialize weights as ones with correct shape
            weights = torch.ones((lower.shape[0], 1), dtype=torch.as_tensor(lower).dtype).numpy()
        else:
            weights = data.get_weight().reshape(-1, 1)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()
        predt_list, loss = self.get_params_loss_censored(
            predt, start_values, lower, upper, requires_grad=True
        )
        grad, hess = self.compute_gradients_and_hessians(loss, predt_list, weights)
        return grad, hess

    def metric_fn(self, predt: np.ndarray, data: xgb.DMatrix):
        lower = data.get_float_info("label_lower_bound")
        upper = data.get_float_info("label_upper_bound")
        if lower.size == 0 and upper.size == 0:
            return super().metric_fn(predt, data)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()
        _, loss = self.get_params_loss_censored(
            predt, start_values, lower, upper, requires_grad=False
        )
        return self.loss_fn, loss

    def get_params_loss_censored(self,
                                 predt: np.ndarray,
                                 start_values: List[float],
                                 lower: np.ndarray,
                                 upper: np.ndarray,
                                 requires_grad: bool = False,
                                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute loss for interval-censored data."""
        predt_arr = predt.reshape(-1, self.n_dist_param)
        # replace nan/inf
        mask = np.isnan(predt_arr) | np.isinf(predt_arr)
        predt_arr[mask] = np.take(start_values, np.where(mask)[1])
        # convert to tensors
        predt_list = [
            torch.tensor(predt_arr[:, i].reshape(-1, 1), requires_grad=requires_grad)
            for i in range(self.n_dist_param)
        ]
        # transform parameters
        params_transformed = [
            fn(predt_list[i]) for i, fn in enumerate(self.param_dict.values())
        ]
        # instantiate distribution
        dist = self.distribution(**dict(zip(self.distribution_arg_names, params_transformed)))
        # compute cdf bounds: convert lower & upper once to tensor with correct dtype
        low = torch.as_tensor(lower, dtype=params_transformed[0].dtype).reshape(-1, 1)
        hi  = torch.as_tensor(upper, dtype=params_transformed[0].dtype).reshape(-1, 1)
        cdf_low = dist.cdf(low)
        cdf_hi = dist.cdf(hi)
        # interval mass & loss
        mass = cdf_hi - cdf_low
        log_density = dist.log_prob(low)
        censored_inds = low != hi
        loss = -torch.sum(torch.log(mass[censored_inds])) - torch.sum(log_density[~censored_inds])
        return predt_list, loss
