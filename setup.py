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
        "xgboost~=1.6.1",
        "torch~=1.13",
        "pyro-ppl~=1.8.5",
        "pandas~=1.5.3",
        "tqdm~=4.0",
        "shap~=0.42.1",
        "seaborn~=0.12.1",
    ],
    extras_require={
        "docs": ["mkdocs", "mkdocstrings[python]", "mkdocs-jupyter"]
    },
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
