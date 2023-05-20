from setuptools import setup, find_packages


__version__ = "0.2.1"

setup(
    name="xgboostlss",
    version=__version__,
    description="XGBoostLSS - An extension of XGBoost to probabilistic forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    maintainer="Alexander März",
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.9",
    install_requires=[
        "xgboost~=1.7.5",
        "torch~=2.0.1",
        "optuna~=3.1.1",
        "properscoring~=0.1",
        "scikit-learn~=1.2.2",
        "numpy~=1.24.3",
        "pandas~=2.0.1",
        "plotnine~=0.12.1",
        "scipy",
        "tqdm",
        "matplotlib",
    ],
    test_suite="tests",
    tests_require=["flake8", "pytest"],
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
    ],

)
