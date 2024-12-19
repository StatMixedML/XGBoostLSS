from setuptools import setup, find_packages

setup(
    name="xgboostlss",
    version="0.4.0",
    description="XGBoostLSS - An extension of XGBoost to probabilistic modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    package_data={"": ["datasets/*.csv"]},
    zip_safe=True,
    python_requires=">=3.10",
    install_requires=[
        "xgboost",
        "torch",
        "pyro-ppl",
        "numpy",
        "pandas",
        "tqdm",
    ],
    extras_require={"docs": ["mkdocs", "mkdocstrings[python]", "mkdocs-jupyter"]},
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
