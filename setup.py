from setuptools import setup, find_packages

setup(
    name="munilagrad",
    version="0.1.0",
    author="Tanish Taywade",
    description="A custom autograd engine and deep learning framework built from scratch",
    packages=find_packages(), # This automatically finds your 'munilagrad' folder
    install_requires=[
        "graphviz" # Automatically installs graphviz when someone installs your package
    ],
    python_requires=">=3.6",
)
