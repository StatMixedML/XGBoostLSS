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


def load_simulated_multivariate_gaussian_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature

    """
    data_path = pkg_resources.resource_stream(__name__, "sim_triv_gaussian.csv")
    data_df = pd.read_csv(data_path)

    return data_df


def load_simulated_multivariate_studentT_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature

    """
    data_path = pkg_resources.resource_stream(__name__, "sim_triv_studentT.csv")
    data_df = pd.read_csv(data_path)

    return data_df


def load_articlake_data():
    """
    Returns the arctic lake sediment data: sand, silt, clay compositions of 39 sediment samples at different water
    depths in an Arctic lake.

    Contains the following columns:
        sand: numeric
            Vector of percentages of sand.
        silt: numeric
            Vector of percentages of silt.
        clay: numeric
            Vector of percentages of clay
        depth: numeric
            Vector of water depths (meters) in which samples are taken.

    Source
    ------
    https://rdrr.io/rforge/DirichletReg/
    """
    data_path = pkg_resources.resource_stream(__name__, "arcticlake.csv")
    data_df = pd.read_csv(data_path)

    return data_df