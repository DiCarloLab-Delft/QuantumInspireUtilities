# QuantumInspireUtilities
Utility functions for Quantum Inspire (QI) 2.0, tailored for the quantum processor Starmon-7.

## Cloning this repository
In order to use the code in this repository, you will need a GitHub account in order to pull the project.
For new users, we recommend to download [GitHub Desktop](https://desktop.github.com/download/), and then clone the repository by using the link https://github.com/DiCarloLab-Delft/QuantumInspireUtilities.git.

## Tested instructions for creating a Python environment compatible with the QI SDK
1. Install Anaconda or Miniconda (lightweight version of Anaconda) in your computer, https://www.anaconda.com/
2. Open Anaconda Prompt (or Terminal in UNIX)
3. Run the following commands
- conda create -n quantuminspire python=3.12  (creates a new conda environment)
- conda activate quantuminspire  (activates the environment)
- pip install quantuminspire  (it is being used in order to login to your QI account)
- pip install qiskit-quantuminspire (the QI SDK)
- pip install jupyterlab
- pip install notebook
- pip install matplotlib
- pip install pandas
- pip install pylatexenc

Once the above steps are done, open Anaconda Prompt (or Terminal in UNIX), activate the environment,
and type "jupyter notebook". This should open a Jupyter notebook page in your browser, and you will be able to use the QI SDK.

In order to create your first quantum circuit using the QI SDK, visit https://qutech-delft.github.io/qiskit-quantuminspire/getting_started/submitting.html
