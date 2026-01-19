"""
Utility functions for solving classically the time-dependent
Schrödinger equation, given an initial quantum state and a system
Hamiltonian.

Authors: Marios Samiotis
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, DensityMatrix, Operator
from qi_utilities.utility_functions.quantum_info import calculate_observable_value

def evolve_quantum_state(quantum_state: QuantumCircuit,
                         hamiltonian: SparsePauliOp,
                         time_step: float):
    """
    This function evolves a (initial) quantum state by applying
    the time evolution unitary operator U(t) = exp(-iHt) on the
    state, for a time_step 't' and a Hamiltonian operator 'H'.

    The initial state 'quantum_state' must be a quantum circuit.
    We follow the common convention hbar=1.

    Args:
        quantum_state (QuantumCircuit):
            The quantum state to be evolved, given as a QuantumCircuit object.
            The user is required to create such a circuit outside of this function.

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

        time_step (float):
            The time step in which we evolve dynamically forward in time
            the quantum_state.
    """

    if type(quantum_state) is not QuantumCircuit:
        raise TypeError(f"Object {quantum_state} must be a QuantumCircuit.")
    
    density_state = DensityMatrix(quantum_state)
    evolution_matrix = PauliEvolutionGate(operator = hamiltonian,
                                          time = time_step)
    evolution_operator = Operator(evolution_matrix)
    evolved_state = density_state.evolve(evolution_operator)

    return evolved_state

def simulate_time_evolution(initial_state: QuantumCircuit,
                            hamiltonian: SparsePauliOp,
                            evolution_times: np.ndarray,
                            observables: list):
    """
    This function takes as inputs an initial quantum state, the system Hamiltonian,
    the evolution times and the list of observables to be extracted, and solves
    the time-dependent Schrödinger equation.
    It returns a dictionary containing the values of each observable in the
    'observables' list, for all time steps in 'evolution_times'.

    The initial state 'initial_state' must be a quantum circuit.
    We follow the common convention hbar=1.

    Args:
        initial_state (QuantumCircuit):
            The quantum state to be evolved, given as a QuantumCircuit object.
            The user is required to create such a circuit outside of this function.

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

        evolution_times (np.ndarray):
            A numpy array containing the discrete time steps for which the simulation
            solves the time-dependent Schrödinger equation.

        observables (list):
            A list containing the observables for which the simulation calculates
            the expectation values of, for each time step within the 'evolution_times'.

            For an n-qubit Pauli string P, where Pauli Pi acts on qubit qi, the order
            in the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit observable string 'IZ', operator Z corresponds to
            qubit q0 while operator I corresponds to qubit q1.
    """

    if type(initial_state) is not QuantumCircuit:
        raise TypeError(f"Object {initial_state} must be a QuantumCircuit.")
    
    dt = evolution_times[1] - evolution_times[0] #use this for the time_interval in noisy evolution
    observables_dict = {}
    for observable in observables:
        observables_dict[observable] = {}
        observables_dict[observable]['list'] = list(observable)
        observables_dict[observable]['values'] = []
    for time_step in evolution_times:
        evolved_state = evolve_quantum_state(initial_state, hamiltonian, time_step)
        # put here the evolution due to a noise channel
        for observable in observables:
            observable_value = calculate_observable_value(evolved_state, observable)
            observables_dict[observable]['values'].append(observable_value)

    return observables_dict