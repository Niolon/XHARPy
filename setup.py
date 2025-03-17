from setuptools import setup, find_packages

setup(
    name="xharpy",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.3",
        "scipy>=1.7.3",
        "pandas>=1.3.5",
        "jax>=0.2.26",
        # Other dependencies
    ],
    # Optional dependencies
    extras_require={
        'gpaw': ['gpaw>=21.6.0'],
        'qe': ['qe>=7.0'],
        'cctbx': ['cctbx>=2021.11']
    }
)