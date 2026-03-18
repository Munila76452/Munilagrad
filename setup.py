from setuptools import setup, find_packages

setup(
    name="munilagrad",
    version="0.1.0",
    author="Tanish Taywade",
    description="A custom autograd engine and deep learning framework built from scratch",
    packages=find_packages(),
    install_requires=[
        "graphviz" # it automatically installs when anyone download munilagrad
    ],
    python_requires=">=3.6",
)
