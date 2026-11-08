"""
Functions for mitigation errors due to readout.

Authors: Jan Hemink
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qi_utilities.utility_functions.readout_correction import (
    extract_ro_assignment_matrix,
    get_ro_corrected_multi_probs,
)
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from numpy.typing import NDArray


def readout_matrix_calibration_circuit(
    num_qubits: int,
    parallel: bool = True,
    name: str = "readout_circuit",
) -> QuantumCircuit:
    """
    Creates a circuit to measure the readout assignment matrix for each individual qubit.
    Still needs to be transpiled for the desired backend, such that the correct physical qubits are
    used.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.
    parallel : bool, optional
        Whether the measurement are performed in parallel, by default True
    name : str, optional
        The name of the circuit, by default "readout_circuit"

    Returns
    -------
    QuantumCircuit
        The readout assignment calibration circuit.
    """
    qubit_list = list(range(num_qubits))

    qc = QuantumCircuit(num_qubits, 2 * num_qubits, name=name)
    qc.reset(qubit_list)

    # measurements at the same time
    if parallel:
        qc.barrier()
        for qubit in qubit_list:
            qc.measure(qubit, qubit)
        qc.barrier()
        qc.reset(qubit_list)
        qc.barrier()
        qc.x(qubit_list)
        qc.barrier()
        for qubit in qubit_list:
            qc.measure(qubit, num_qubits + qubit)

    # measurements separated
    else:
        qc.barrier(0)
        for qubit in qubit_list:
            qc.measure(qubit, qubit)
            qc.barrier([qubit, (qubit + 1) % num_qubits])
            qc.reset(qubit)

        for qubit in qubit_list:
            qc.x(qubit)
            qc.barrier(qubit)
            qc.measure(qubit, num_qubits + qubit)
            if (qubit + 1) < num_qubits:
                qc.barrier([qubit, (qubit + 1) % num_qubits])
            else:
                qc.barrier(qubit)

    return qc


def analyse_readout_matrix_circuit(
    ro_shots: list[str],
    qubit_list: list[int],
) -> dict[str, NDArray]:
    """
    Analyses the data obtained from the readout assignment calibration circuit and returns the
    readout assignment matrix for each individual qubit.

    Parameters
    ----------
    ro_shots : list[str]
        The raw data shots from the readout correction calibration circuit.
    qubit_list : list[int]
        Ordered list of qubits.

    Returns
    -------
    dict[str, NDArray]
        The readout assignment matrices for each qubit.
    """
    ro_matrices = {}
    num_qubits = len(qubit_list)

    for idx, qubit in enumerate(qubit_list):
        qubit_shots = [shot[(-1 - idx - num_qubits)] + shot[(-1 - idx)] for shot in ro_shots]
        ro_matrices[f"Qubit {qubit}"] = extract_ro_assignment_matrix(qubit_shots, [qubit])

    return ro_matrices


def apply_readout_assignment_correction(
    raw_probs: dict[str, float],
    ro_matrices: dict[str, NDArray],
    qubit_list: list[int],
) -> dict[str, float]:
    """
    Performs readout assignment correction. Combines the readout assignment matrices of the
    individual qubits, then obtains the corrected probabilities.

    Parameters
    ----------
    raw_probs : dict[str, float]
        The raw probabilities extracted from the desired circuit.
    ro_matrices : dict[str, NDArray]
        The readout assignment matrices for each individual qubit.
    qubit_list : list[int]
        Ordered list of qubits.

    Returns
    -------
    dict[str, float]
        The readout assignment error mitigated probabilities.

    See Also
    --------
    `readout_correction_circuit`
        For creating a circuit to calibrate the readout assignment matrices.
    `analyse_readout_matrix_circuit`
        To obtain the readout assignment matrices from the calibration circuit
    """
    if len(qubit_list) == 1:
        ro_matrix = ro_matrices[f"Qubit {qubit_list[0]}"]
    else:
        ro_matrix = np.kron(ro_matrices[f"Qubit {qubit_list[1]}"],ro_matrices[f"Qubit {qubit_list[0]}"])
        for qubit in qubit_list[2:]:
            ro_matrix = np.kron(ro_matrices[f"Qubit {qubit}"], ro_matrix)

    # perform ro_correction for each qubit individually
    return get_ro_corrected_multi_probs(
        raw_data_probs=[raw_probs],
        ro_assignment_matrix=ro_matrix,
        qubit_list=qubit_list,
    )[0]
