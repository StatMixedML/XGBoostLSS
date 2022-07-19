import numpy as np
from scipy.stats import norm


def get_scale_shares(pred_quantiles, quant_sel):
    n_samples = pred_quantiles.shape[0]
    quantile = float(quant_sel.split("_")[1])

    # compare estimated quantiles with the theoretical quantiles
    scale_1_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=1), atol=1)
    scale_3_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=3), atol=1)
    scale_5_filter = np.isclose(pred_quantiles[quant_sel], norm.ppf(quantile, loc=10, scale=5), atol=1)

    share_s1 = pred_quantiles.loc[scale_1_filter, quant_sel].count() / n_samples
    share_s3 = pred_quantiles.loc[scale_3_filter, quant_sel].count() / n_samples
    share_s5 = pred_quantiles.loc[scale_5_filter, quant_sel].count() / n_samples
    return share_s1, share_s3, share_s5
