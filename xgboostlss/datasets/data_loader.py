import pkg_resources
import pandas as pd
import numpy as np
import torch
import torchlambertw.distributions as tlwd


def gen_features(n_samples: int, n_dims: int) -> pd.DataFrame:
    """Generates a feature DataFrame with uniform(0, 1) in each dimension. Use first as 'true'."""
    rng = np.random.RandomState(seed=n_samples)
    print(rng)
    X = pd.DataFrame(rng.uniform(size=(n_samples, n_dims)))
    X.columns = ["x_true"] + ["x_noise" + str(k + 1) for k in range(1, n_dims)]
    return X


def gen_gaussian_data(n_samples: int) -> pd.DataFrame:
    X = gen_features(n_samples, n_dims=11)
    loc_true = pd.Series([10.0] * n_samples, name="loc")
    scale_true = (
        1.0
        + 4 * ((X["x_true"] > 0.3) & (X["x_true"] < 0.5)).astype(float)
        + 2 * (X["x_true"] > 0.7).astype(float)
    )
    scale_true.name = "scale"

    df = pd.concat([X, loc_true, scale_true], axis=1)
    torch.manual_seed(n_samples)
    df["y"] = (
        torch.distributions.Normal(
            loc=torch.tensor(df["loc"].values), scale=torch.tensor(df["scale"].values)
        )
        .sample([1])
        .numpy()
        .ravel()
    )
    return df


def gen_tail_lambertw_gaussian_data(n_samples: int) -> pd.DataFrame:
    X = gen_features(n_samples, n_dims=11)
    loc_true = pd.Series([10.0] * n_samples, name="loc")
    scale_true = (
        1.0
        + 4 * ((X["x_true"] > 0.3) & (X["x_true"] < 0.5)).astype(float)
        + 2 * (X["x_true"] > 0.7).astype(float)
    )
    scale_true.name = "scale"

    tailweight_true = (
        0.0
        + 0.3 * ((X["x_true"] > 0.2) & (X["x_true"] < 0.4)).astype(float)
        + 0.1 * (X["x_true"] > 0.9).astype(float)
    )
    tailweight_true.name = "tailweight"

    df = pd.concat([X, loc_true, scale_true, tailweight_true], axis=1)

    torch.manual_seed(n_samples)
    distr = tlwd.TailLambertWNormal(
        loc=torch.tensor(df["loc"].values),
        scale=torch.tensor(df["scale"].values),
        tailweight=torch.tensor(df["tailweight"].values),
    )
    df["y"] = distr.sample([1]).numpy().ravel()
    df["q5"] = distr.icdf(torch.tensor([0.05]))
    df["q95"] = distr.icdf(torch.tensor([0.95]))
    return df


def load_simulated_tail_lambertw_gaussian_data() -> pd.DataFrame:
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    all_df = gen_tail_lambertw_gaussian_data(n_samples=10000)
    train_df, test_df = all_df.iloc[:7000], all_df.iloc[7000:]
    return train_df, test_df


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
