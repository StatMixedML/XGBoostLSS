from setuptools import setup, find_packages


import re

_VERSION_FILE = "xgboostlss/_version.py"
verstrline = open(_VERSION_FILE, "rt").read()
_VERSION = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(_VERSION, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (_VERSION_FILE,))


setup(
    name="xgboostlss",
    version=verstr,
    description="XGBoostLSS - An extension of XGBoost to probabilistic modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Alexander M"{a}rz',
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    package_data={"": ["datasets/*.csv"]},
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=[
        "xgboost>=1.6.1",
        "torch>=2.0.1",
        "pyro-ppl>=1.5.0",
        "optuna>=3.0.0",
        "properscoring>=0.1",
        "scikit-learn>=1.0.2",
        "numpy>=1.23.0",
        "pandas>=2.0.3",
        "plotnine>=0.10.0",
        "statsmodels>=0.14.0",
        "scipy>=1.0.0",
        "seaborn>=0.13.0",
        "torchlambertw @ git+ssh://git@github.com/gmgeorg/torchlambertw.git#egg=torchlambertw-0.0.3",
        "tqdm>=4.0.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={"docs": ["mkdocs", "mkdocstrings[python]", "mkdocs-jupyter"]},
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
