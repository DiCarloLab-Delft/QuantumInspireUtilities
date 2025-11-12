import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, DensityMatrix, Operator
from qi_utilities.utility_functions.quantum_info import calculate_observable_value

def evolve_quantum_state(quantum_state: QuantumCircuit,
                         hamiltonian: SparsePauliOp,
                         time_step: float):
    """
    Initial state must be a quantum circuit

    Applies the unitary U(t) = exp(-iHt) on a quantum state
    We follow the convention hbar=1.
    """
    density_state = DensityMatrix(quantum_state)
    evolution_matrix = PauliEvolutionGate(operator = hamiltonian,
                                          time = time_step)
    evolution_operator = Operator(evolution_matrix)

    evolved_state = density_state.evolve(evolution_operator)

    return evolved_state

def simulate_time_evolution(initial_state: QuantumCircuit,
                            hamiltonian: SparsePauliOp,
                            evolution_times: np.array,
                            observables: list):
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