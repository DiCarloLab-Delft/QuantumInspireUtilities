import numpy as np
from qiskit.quantum_info import SparsePauliOp

def obtain_binary_list(nr_qubits: int):
    binary_list = []
    for binary_str_idx in range(2**nr_qubits):
        binary_list.append(np.binary_repr(binary_str_idx, nr_qubits))
    return binary_list

def get_multi_qubit_counts(shots,
                           nr_qubits: int):
    """
    Returns a list with containing entries of all count dictionaries for
    each midcircuit measurement block.
    e.g. total_counts[0] contains the counts dictionary for the very first
    measurement block.
    """
    binary_list = obtain_binary_list(nr_qubits)

    total_counts = []

    mid_circuit_blocks_nr = int(len(shots[0]) / nr_qubits)

    for mcm_block_idx in range(mid_circuit_blocks_nr):

        counts_dict = {}
        for entry in binary_list:
            counts_dict[entry] = 0

        for shot_idx in range(len(shots)):

            reversed_shots = shots[shot_idx][::-1]

            binary_string_reversed = reversed_shots[mcm_block_idx*nr_qubits:(mcm_block_idx+1)*nr_qubits]
            # in order to ensure that len(binary_string) == nr_qubits,
            binary_string = binary_string_reversed[::-1]
            binary_string = np.binary_repr(int(binary_string, 2), nr_qubits)
            counts_dict[binary_string] += 1

        total_counts.append(counts_dict)
    return total_counts

def get_multi_qubit_prob(counts):

    probabilities = []

    nr_shots = 0
    for entry in counts[0]:
        nr_shots += counts[0][entry]

    for entry_idx in range(len(counts)):
        prob_dict = {}
        for entry in counts[entry_idx]:
            prob_dict[entry] = counts[entry_idx][entry] / nr_shots
        probabilities.append(prob_dict)

    return probabilities


def extract_observable_values_Z_basis(probabilities: list[dict],
                                      observable: list,
                                      nr_qubits: int):
    """
    'probabilitites' : list
                     must be a list of dictionaries of the form
                     probabilitities = [{'00': 1000, '01': 1000, '10': 1000, '11': 1000},
                                        ...]
    """

    binary_list = obtain_binary_list(nr_qubits)

    observable_values = []
    observable_matrix = np.real(SparsePauliOp([observable]).to_matrix())

    for entry_idx in range(len(probabilities)):
        observable = 0
        for binary_idx in range(len(binary_list)):
            observable += probabilities[entry_idx][binary_list[binary_idx]] * observable_matrix[binary_idx][binary_idx]
        observable_values.append(observable)

    return observable_values
