# QuantumInspireUtilities
This repository contains utility functions and demonstration notebooks that supplement the [Quantum Inspire 2.0](https://www.quantum-inspire.com/) Python SDK, tailored for the superconducting backends of the platform, hosted in the [DiCarlo lab](https://qutech.nl/lab/dicarlo-lab-welcome/) in QuTech.

It is primarily intended for researchers, students, and developers who use Quantum Inspire’s superconducting backends and want reusable utilities and worked examples beyond the core SDK.

Useful links relating to our superconducting backends:
1. [Tuna Backends Operational Specifics](https://www.quantum-inspire.com/kbase/tuna-operational-specifics/)
2. [Backend Performance Dashboards](https://monitoring.qutech.support/public-dashboards/c494a21fb6b7405f850ab8f340f798ef?orgId=1&refresh=10s) [live updates]
3. [Join our Slack community!](https://join.slack.com/t/qisuperconducting/shared_invite/zt-35o7zitdh-_9QPmB53hhLy12Eat5gwWA)

## 1. Installation instructions

### 1.1. Beginner-friendly installation
Note: this installation method is typically not recommended, but is nevertheless suggested due to its relative simplicity. In principle it avoids installing and using pipx, which some users have experienced difficulties in doing so.

1. [Install Anaconda](https://www.anaconda.com/) or Miniconda (lightweight version of Anaconda) in your computer
2. Open Anaconda Prompt (or Terminal in UNIX)
3. Run the following commands
- conda create -n quantuminspire python=3.12  (creates a new conda environment)
- conda activate quantuminspire  (activates the environment)
- pip install quantuminspire
- pip install qi-utilities (installs this repository)
- pip install jupyterlab
- pip install notebook

Installing quantuminspire within the conda environment restricts the command 'qi login' to be recognized and used only within the created environment.

### 1.2. Advanced installation (recommended)

1. [Install pipx](https://pipx.pypa.io/stable/installation/) (used when installing quantuminspire package)
2. [Install quantuminspire repository](https://github.com/QuTech-Delft/quantuminspire?tab=readme-ov-file) (used for login)
3. [Install Anaconda](https://www.anaconda.com/) or Miniconda (lightweight version of Anaconda) in your computer
4. Open Anaconda Prompt (or Terminal in UNIX)
5. Run the following commands
- conda create -n quantuminspire python=3.12  (creates a new conda environment)
- conda activate quantuminspire  (activates the environment)
- pip install qi-utilities (installs this repository)
- pip install jupyterlab
- pip install notebook

Note: in order to run the method backend.coupling_map.draw(), you will need to [install Graphviz](https://graphviz.org/download/#executable-packages) in your computer. Make sure during installation to add Graphviz to the system PATH, so that your Python environment can recognize it.

## 2. Cloning and installing from source
In order to install this repository from source, you will need a GitHub account to be able to pull the project.
For new users, we recommend downloading [GitHub Desktop](https://desktop.github.com/download/), and then cloning the repository by using the link https://github.com/DiCarloLab-Delft/QuantumInspireUtilities.git.

After creating a working Python environment (see instructions above), open Anaconda Prompt to activate the environment, navigate to the repository directory, and use the following pip command to [install](https://docs.python.org/3/installing/index.html),

python -m pip install -e .

The '-e' flag makes the package configurable.

In order to create your first quantum circuit using the Quantum Inspire SDK, visit https://qutech-delft.github.io/qiskit-quantuminspire/getting_started/submitting.html.

## 3. TODOs
[2026-01-20] Current list of targeted upcoming new features:

* Create utility functions and a demonstration notebook guide on variational quantum algorithms.
