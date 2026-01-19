"""
Utility functions for processing the raw data shots that have been obtained
from circuits that may also contain mid-circuit measurements.

NOTE: for all of these functions to be useful, the user must request the memory
of a job prior to executing it,
e.g. job = backend.run(qc, shots, memory = True)

Authors: Marios Samiotis
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.result.result import Result
from qiskit.quantum_info import SparsePauliOp

def obtain_binary_list(nr_qubits: int):
    """
    This function returns an ordered list of binary numbers as a function
    of nr_qubits.

    e.g. for nr_qubits = 2, binary_list = ['00', '01', '10', '11']

    Args:
        nr_qubits (int):
            The total number of qubits for which the binary list will be
            created for.
    """

    binary_list = []
    for binary_str_idx in range(2**nr_qubits):
        binary_list.append(np.binary_repr(binary_str_idx, nr_qubits))
    return binary_list

def get_raw_data(qc: QuantumCircuit,
                 result: Result,
                 circuit_nr: int = None):
    """
    This function returns the raw_data for each measurement block within
    a given quantum circuit 'qc' in an organized way.

    Args:
        qc (QuantumCircuit):
            The quantum circuit object.

        result (Result):
            The result of a job (project), as returned from
            result = job.result()

        circuit_nr (int):
            The circuit number within a job, since a job can contain
            multiple quantum circuits.
            Defaults to None for a job with a single quantum circuit.
    """

    bit_register_size = qc.num_clbits
    raw_data = result.get_memory(circuit_nr)
    for entry in range(len(raw_data)):
        additional_len = bit_register_size - len(raw_data[entry])
        for i in range(additional_len):
            raw_data[entry] = '0' + raw_data[entry]

    return raw_data

def get_multi_counts(raw_data_shots: list,
                     nr_qubits: int):
    """
    This function returns a list containing entries of all count dictionaries
    for each measurement block within one quantum circuit.
    e.g. total_counts[0] contains the counts dictionary for the very first
    measurement block of a given quantum circuit.

    Args:
        raw_data_shots (list):
            The raw data shots returned from quantum circuits containing
            a single or multiple mid-circuit measurement blocks.

            For a quantum circuit containing M number of mid-circuit measurements,
            and executing it for a number of N shots, len(raw_data_shots) = N,
            while each entry of the list contains a bitstring of size M.

            The convention followed in each bitstring for a bit register of size K
            is 'cK-1,cK-2,...,c2,c1,c0', meaning that the rightmost bit corresponds
            to the very first bit in the bit register.

        nr_qubits (int):
            The number of qubits of the original quantum circuit.
    """

    binary_list = obtain_binary_list(nr_qubits)
    mid_circuit_blocks_nr = int(len(raw_data_shots[0]) / nr_qubits)

    total_counts = []
    for mcm_block_idx in range(mid_circuit_blocks_nr):

        counts_dict = {}
        for entry in binary_list:
            counts_dict[entry] = 0

        for shot_idx in range(len(raw_data_shots)):
            reversed_shots = raw_data_shots[shot_idx][::-1]

            binary_string_reversed = reversed_shots[mcm_block_idx*nr_qubits:(mcm_block_idx+1)*nr_qubits]
            # in order to ensure that len(binary_string) == nr_qubits,
            binary_string = binary_string_reversed[::-1]
            binary_string = np.binary_repr(int(binary_string, 2), nr_qubits)
            counts_dict[binary_string] += 1

        total_counts.append(counts_dict)
    return total_counts

def get_multi_probs(raw_data_counts: list[dict]):
    """
    This function takes in a list of the total measurement counts for each
    measurement block in a quantum circuit, and computes
    the probabilities for each block.

    Args:
        raw_data_counts (list[dict]):
            The total measurement counts for each (mid-circuit) measurement
            block in a given quantum circuit. The first entry of the list
            corresponds to the measurement counts of the first measurement
            block, while the last entry corresponds to the counts of the
            last measurement block.
    """

    nr_shots = 0
    for entry in raw_data_counts[0]:
        nr_shots += raw_data_counts[0][entry]

    probabilities = []
    for entry_idx in range(len(raw_data_counts)):
        prob_dict = {}
        for entry in raw_data_counts[entry_idx]:
            prob_dict[entry] = raw_data_counts[entry_idx][entry] / nr_shots
        probabilities.append(prob_dict)

    return probabilities


def observable_expectation_values_Z_basis(probabilities: list[dict],
                                          observable: str):
    """
    This function calculates the expectation values of an observable
    in the Z basis, given a list of measurement probabilities.
    In case the user wants to calculate the expectation value of an observable
    containing Paulis X and Y, first the appropriate pre-measurement rotations
    must be applied in the quantum circuit so that those are projected to the
    Z basis.

    Args:
        probabilitites (list[dict]):
            A list of dictionaries each containing the measurement probabilities
            of a certain measurement block.
            e.g. probabilitities = [{'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25},
                                    ...]

        observable (str):
            The observable for which this function calculates the expectation values of.
            It must be expressed in the Z basis, containing no Paulis X or Y.

            For an n-qubit Pauli string P, where Pauli Pi acts on qubit qi, the order
            in the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit observable string 'IZ', operator Z corresponds to
            qubit q0 while operator I corresponds to qubit q1.
    """

    if 'X' in observable or 'Y' in observable:
        raise ValueError(f"Observable {observable} must not contain Paulis X or Y.")
    
    nr_qubits = len(observable)
    binary_list = obtain_binary_list(nr_qubits)

    observable_values = []
    observable_matrix = np.real(SparsePauliOp([observable]).to_matrix())

    for entry_idx in range(len(probabilities)):
        observable = 0
        for binary_idx in range(len(binary_list)):
            observable += probabilities[entry_idx][binary_list[binary_idx]] * observable_matrix[binary_idx][binary_idx]
        observable_values.append(observable)

    return observable_values