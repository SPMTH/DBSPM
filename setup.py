from setuptools import setup

setup(
    name="DBSPM",
    version="0.1a",
    packages=["pydbspm"],
    install_requires=[
        "numpy<2",
        "scipy",
        "mpi4py",
        "dftd3",
        "ase==3.22.1",
        "pandas",
        "matplotlib",
        "tricubic",
    ],
    author="E. Ventura-Macias",
    scripts=["dbspm"],
    author_email="emilianoventura@outlook.com",
    description="Density-Based Scanning-Probe Method",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],
)
