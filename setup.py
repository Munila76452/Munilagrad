from setuptools import setup, find_packages

# This reads your README.md to use as the long description on PyPI
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="munilagrad", # This is the name people will type in pip install
    version="0.1.0",   # You must change this number every time you push an update
    author="Tanish Taywade",
    description="A custom autograd engine and deep learning framework built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Munila76452/Munilagrad", # Link to your GitHub
    packages=find_packages(), 
    install_requires=[
        "graphviz" # it automatically installs when anyone downloads munilagrad
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
