import pkg_resources
import pandas as pd


def load_simulated_data():
    """Returns train/test dataframe of a similated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    
    train_df = pkg_resources.resource_stream(__name__, 'datasets/train_sim.csv')
    test_df = pkg_resources.resource_stream(__name__, 'datasets/test_sim.csv')
    return pd.read_csv(train_df), pd.read_csv(test_df)