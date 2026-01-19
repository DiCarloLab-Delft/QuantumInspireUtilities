"""
Utility functions for synthesizing a Trotterization quantum circuit.

Authors: Marios Samiotis
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qi_utilities.utility_functions.circuit_modifiers import apply_pre_measurement_rotations

def apply_trotter_block(qc: QuantumCircuit,
                        hamiltonian: SparsePauliOp,
                        trotter_order: int,
                        time_step: float):
    """
    This function applies a Trotter block on a given quantum circuit
    'qc'.
    We follow the common convention hbar=1.

    Args:
        qc (QuantumCircuit):
            The quantum circuit object.

        hamiltonian (SparsePauliOp):
            The Hamiltonian operator describing the dynamics of a given
            quantum system, written in the Pauli basis.
            It should be given in units of [Hz].

            In general, a Hamiltonian operator is written as
            H = sum_{j} a_j * P_j, where a_j are complex coefficients in units of [Hz],
            and P_j are n-qubit Pauli operators (else referred to as 'Pauli strings').

            For an n-qubit Pauli operator P in the Hamiltonian, with Pauli Pi acting
            on qubit qi, the ordering of the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit Pauli string 'YX', operator X corresponds
            to qubit q0, while operator Y corresponds to qubit q1.

        trotter_order (int):
            The trotterization order. For more info, visit
            https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas

        time_step (float):
            The time step in which we evolve dynamically forward in time
            the quantum state encoded in the quantum circuit 'qc'.
    """
    hamiltonian = hamiltonian[::-1]
    for entry in range(len(hamiltonian)):
        pauli_string = hamiltonian[entry].paulis.to_labels()[0]

        string_to_list = list(pauli_string)
        target_qubits = []
        qubit_counter = 0
        for index in reversed(range(len(string_to_list))):
            if string_to_list[index] == 'I':
                pass
            else:
                target_qubits.append(qubit_counter)
            qubit_counter += 1

        unitary_label = 'Trotter block,' + f' Pauli: {pauli_string}' + \
                        f'\nn = {trotter_order},' + f' Time = {time_step*1e9:.2f} ns'
        unitary_gate = PauliEvolutionGate(operator = hamiltonian[entry],
                                          time = time_step / trotter_order,
                                          label = unitary_label)
        qc.append(unitary_gate, target_qubits)

def construct_trotterization_circuit(initial_state: str,
                                     measured_observable: str,
                                     hamiltonian: SparsePauliOp,
                                     trotter_order: int,
                                     evolution_times: np.ndarray,
                                     time_step: float,
                                     midcircuit_measurement: bool = False):
    """
    This function uses the 'trotter_block' function contained in this module
    to create the Trotterization quantum circuit from an initial state.
    We follow the common convention hbar=1.

    Args:
        initial_state (str):
            A bitstring specifying the initial state. The order in the bitstring
            is 'qn-1,qn-2,...,q2,q1,q0'. The string can only contain '0' or '1'.
            e.g. for a quantum circuit with 3 qubits, initial_state = '100' will
            initialize qubits q0 and q1 in the state |0>, and qubit q2 in the state
            |1>.

        measured_observable (str):
            The observable whose expectation value we estimate at the end of
            the Trotterization algorithm.

            For an n-qubit Pauli string P, where Pauli Pi acts on qubit qi, the order
            in the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit observable string 'YX', qubit q0 is measured
            in the X basis, while qubit q1 is measured in the Y basis.

        hamiltonian (SparsePauliOp):
            The Hamiltonian operator describing the dynamics of a given
            quantum system, written in the Pauli basis.
            It should be given in units of [Hz].

            In general, a Hamiltonian operator is written as
            H = sum_{j} a_j * P_j, where a_j are complex coefficients in units of [Hz],
            and P_j are n-qubit Pauli operators (else referred to as 'Pauli strings').

            For an n-qubit Pauli operator P in the Hamiltonian, with Pauli Pi acting
            on qubit qi, the ordering of the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit Pauli string 'YX', operator X corresponds
            to qubit q0, while operator Y corresponds to qubit q1.

        trotter_order (int):
            The trotterization order. For more info, visit
            https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas

        evolution_times (np.ndarray):
            A numpy array containing the discrete time steps for which the simulation
            solves the time-dependent Schrödinger equation.

        time_step (float):
            The time step in which we evolve dynamically forward in time
            the quantum state encoded in the quantum circuit 'qc'.

        midcircuit_measurement (bool):
            A flag which should be set to 'True' if we are utilizing the mid-circuit
            functionality when constructing the Trotterization algorithm.
    """

    nr_qubits = len(initial_state)
    if midcircuit_measurement == True:
        qc = QuantumCircuit(nr_qubits, nr_qubits*len(evolution_times))
    else:
        qc = QuantumCircuit(nr_qubits, nr_qubits)
    for idx in range(nr_qubits):
        qc.reset(idx)
    qc.barrier()

    for idx in range(len(initial_state)):
        if initial_state[idx] == '1':
            qc.x((nr_qubits-1) - idx)
    qc.barrier()

    for repetition in range(1, trotter_order+1):
        apply_trotter_block(qc, hamiltonian, trotter_order, evolution_times[time_step])
    qc.barrier()

    if midcircuit_measurement == True:
        apply_pre_measurement_rotations(qc, measured_observable, [nr_qubits*time_step, nr_qubits*time_step + 1])
    else:
        apply_pre_measurement_rotations(qc, measured_observable)
    qc.barrier()

    return qc