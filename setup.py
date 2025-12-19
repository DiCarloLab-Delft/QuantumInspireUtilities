from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()
    
def license():
    with open('LICENSE') as f:
        return f.read()

setup(
    name='qi_utilities',
    version='0.1.0',
    author='DiCarlo Lab at QuTech',
    author_email='secr-qutech@tudelft.nl',
    maintainer='Marios Samiotis',
    maintainer_email='m.samiotis@tudelft.nl',
    description=(
        '''Package library with utility functions and demonstration
        notebooks especially suited for the superconducting hardware
        backends of the cloud quantum computing platform Quantum Inspire.

        Install using the pip command in the parent directory:
            python -m pip install -e .'''
    ),
    long_description=readme(),
    url='https://github.com/DiCarloLab-Delft/QuantumInspireUtilities',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3 :: Only',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Programming Language :: Python :: 3.12',
                 'Topic :: Scientific/Engineering'],
    keywords='qiskit quantuminspire quantum computing superconducting',
    license=license(),
    packages=find_packages(),
    install_requires=list(open('requirements.txt')
                          .read()
                          .strip()
                          .split('\n')),
    tests_require=['pytest'],
    zip_safe=False)