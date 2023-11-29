from setuptools import setup, find_packages

setup(
    name="xgboostlss",
    version="0.4.0",
    description="XGBoostLSS - An extension of XGBoost to probabilistic forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.9",
    install_requires=[
        "xgboost~=2.0.2",
        "torch~=2.1.1",
        "pyro-ppl~=1.8.6",
        "optuna~=3.4.0",
        "properscoring~=0.1",
        "scikit-learn~=1.3.2",
        "numpy~=1.26.2",
        "pandas~=2.1.3",
        "plotnine~=0.12.4",
        "scipy~=1.11.4",
        "seaborn~=0.13.0",
        "tqdm~=4.66.1",
        "matplotlib~=3.8.2",
        "ipython~=8.18.1",
    ],
    extras_require={
        "docs": ["mkdocs", "mkdocstrings[python]", "mkdocs-jupyter"]
    },
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
