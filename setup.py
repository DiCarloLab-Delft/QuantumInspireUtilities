from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent

def read_file(filename):
    return (here / filename).read_text(encoding="utf-8")

def read_requirements(filename="requirements.txt"):
    return [
        line.strip()
        for line in (here / filename).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="qi_utilities",
    version="0.2.1",
    author="DiCarlo Lab at QuTech",
    author_email="secr-qutech@tudelft.nl",
    maintainer="Marios Samiotis",
    maintainer_email="m.samiotis@tudelft.nl",
    description=(
        "Package library with utility functions and demonstration "
        "notebooks especially suited for the superconducting hardware "
        "backends of the cloud quantum computing platform Quantum Inspire."
    ),
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/DiCarloLab-Delft/QuantumInspireUtilities",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit quantuminspire quantum computing superconducting",
    license=read_file("LICENSE"),
    packages=find_packages(),
    install_requires=read_requirements(),
    tests_require=["pytest"],  # legacy, see note below
    zip_safe=False,
)