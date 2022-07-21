from setuptools import setup, find_packages


def get_version():
    from os.path import abspath, dirname, join
    abspath = abspath(__file__)
    abspath_dir = dirname(abspath)
    pyproject_path = join(abspath_dir, '..', 'pyproject.toml')

    with open(pyproject_path, 'r') as f:
        for line in f:
            if 'version' in line:
                return line.split('=')[1].strip().replace('"', '').replace("'", "")


version = get_version()

setup(
    name="xgboostlss",
    version=version,
    description="XGBoostLSS - An extension of XGBoost to probabilistic forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    maintainer="Alexander März, Sebastian Cattes",
    author_email="alex.maerz@gmx.net, sebastian.cattes@inwt-statistics.de",
    url="https://github.com/StatMixedML/XGBoostLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.8, <3.11",
    install_requires=[
        "xgboost>=1.6.1",
        "optuna>=2.10.1",
        "torch>=1.12.0",
        "shap>=0.41.0",
        "numpy>=1.22.4",
        "pandas>=1.4.3",
        "scipy>=1.8.1",
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
