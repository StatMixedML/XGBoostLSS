"""XGBoostLSS - An extension of XGBoost to probabilistic forecasting"""


def get_version():
    from os.path import abspath, dirname, join
    abspath = abspath(__file__)
    abspath_dir = dirname(abspath)
    pyproject_path = join(abspath_dir, '..', 'pyproject.toml')

    with open(pyproject_path, 'r') as f:
        for line in f:
            if 'version' in line:
                return line.split('=')[1].strip().replace('"', '').replace("'", "")


__version__ = get_version()
