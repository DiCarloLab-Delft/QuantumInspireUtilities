from qiskit.quantum_info import DensityMatrix, Pauli

def calculate_observable_value(density_state: DensityMatrix,
                                observable: str):
    observable_operator = Pauli(observable)
    observable_expectation_value = density_state.expectation_value(observable_operator)

    return observable_expectation_value