# QuantumInspireUtilities
Utility functions that supplement the [Quantum Inspire 2.0](https://www.quantum-inspire.com/) Python SDK, tailored for the superconducting backends Starmon-7 and Tuna-5.

## Cloning this repository
In order to use the functions in this repository, you will need a GitHub account in order to pull the project.
For new users, we recommend downloading [GitHub Desktop](https://desktop.github.com/download/), and then cloning the repository by using the link https://github.com/DiCarloLab-Delft/QuantumInspireUtilities.git.

## Tested instructions for creating a Python environment compatible with the QI SDK
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

Once the above steps are done, open Anaconda Prompt (or Terminal in UNIX), activate the environment,
and type 'jupyter notebook'. This should open a Jupyter notebook page in your browser, and you will be able to use the QI SDK.

In order to create your first quantum circuit using the Quantum Inspire SDK, visit https://qutech-delft.github.io/qiskit-quantuminspire/getting_started/submitting.html.
