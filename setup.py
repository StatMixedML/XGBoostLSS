from setuptools import setup, find_packages

version = "0.1.1"

setup(
    name="xgboost-lss",
    version=version,
    description="XGBoostLSS - An extension of XGBoost to probabilistic forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März, Sebastian Cattes",
    author_email="alex.maerz@gmx.net, sebastian.cattes@inwt-statistics.de",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.7, <4",
    install_requires=[
        "xgboost>=1.6.1",
        "optuna>=2.10.1",
        "torch>=1.12.0",
        "shap>=0.41.0",
        "numpy>=1.21.6",
        "pandas>=1.3.5",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.2",
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
