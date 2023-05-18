import pkg_resources
import pandas as pd


def load_simulated_gaussian_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    train_path = pkg_resources.resource_stream(__name__, "gaussian_train_sim.csv")
    train_df = pd.read_csv(train_path)

    test_path = pkg_resources.resource_stream(__name__, "gaussian_test_sim.csv")
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def load_simulated_studentT_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    train_path = pkg_resources.resource_stream(__name__, "studentT_train_sim.csv")
    train_df = pd.read_csv(train_path)

    test_path = pkg_resources.resource_stream(__name__, "studentT_test_sim.csv")
    test_df = pd.read_csv(test_path)

    return train_df, test_df
