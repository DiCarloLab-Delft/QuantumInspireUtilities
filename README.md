# QuantumInspireUtilities
Utility functions that supplement the [Quantum Inspire 2.0](https://www.quantum-inspire.com/) Python SDK, tailored for the superconducting backends of the platform, hosted in the [DiCarlo lab](https://qutech.nl/lab/dicarlo-lab-welcome/) in QuTech.

## 1. Tested instructions for creating a Python environment compatible with the QI SDK

### 1.1. Beginner-friendly installation
Note: this installation method is typically not recommended, but is nevertheless suggested due to its relative simplicity. In principle it avoids installing and using pipx, which some users have experienced difficulties in doing so.

1. [Install Anaconda](https://www.anaconda.com/) or Miniconda (lightweight version of Anaconda) in your computer
2. Open Anaconda Prompt (or Terminal in UNIX)
3. Run the following commands
- conda create -n quantuminspire python=3.12  (creates a new conda environment)
- conda activate quantuminspire  (activates the environment)
- pip install quantuminspire
- pip install qiskit-quantuminspire
- pip install jupyterlab
- pip install notebook
- pip install matplotlib
- pip install pandas
- pip install pylatexenc

Installing quantuminspire within the conda environment restricts the command 'qi login' to be recognized and used only within the created environment.

### 1.2. Advanced installation

1. [Install pipx](https://pipx.pypa.io/stable/installation/) (used when installing quantuminspire package)
2. [Install quantuminspire repository](https://github.com/QuTech-Delft/quantuminspire?tab=readme-ov-file) (used for login)
3. [Install Anaconda](https://www.anaconda.com/) or Miniconda (lightweight version of Anaconda) in your computer
4. Open Anaconda Prompt (or Terminal in UNIX)
5. Run the following commands
- conda create -n quantuminspire python=3.12  (creates a new conda environment)
- conda activate quantuminspire  (activates the environment)
- pip install qiskit-quantuminspire
- pip install jupyterlab
- pip install notebook
- pip install matplotlib
- pip install pandas
- pip install pylatexenc

Note: in order to run the method backend.coupling_map.draw(), you will need to [install Graphviz](https://graphviz.org/download/#executable-packages) in your computer. Make sure during installation to add Graphviz to the system PATH, so that your Python environment can recognize it.

## 2. Cloning and installing this repository
In order to use the functions in this repository, you will need a GitHub account to be able to pull the project.
For new users, we recommend downloading [GitHub Desktop](https://desktop.github.com/download/), and then cloning the repository by using the link https://github.com/DiCarloLab-Delft/QuantumInspireUtilities.git.

After creating a working Python environment (see instructions above), open Anaconda Prompt to activate the environment, navigate to the repository directory, and use the following pip command to [install](https://docs.python.org/3/installing/index.html),

python -m pip install -e .

Once the above steps are done, open Anaconda Prompt (or Terminal in UNIX), activate the environment,
and type 'jupyter notebook'. This should open a Jupyter notebook page in your browser, and you will be able to use the QI SDK.

In order to create your first quantum circuit using the Quantum Inspire SDK, visit https://qutech-delft.github.io/qiskit-quantuminspire/getting_started/submitting.html.
