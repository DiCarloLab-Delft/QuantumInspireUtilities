"""
Utility functions for generic quantum information calculations.

Authors: Marios Samiotis
"""

from qiskit.quantum_info import DensityMatrix, Pauli

def calculate_observable_value(density_state: DensityMatrix,
                               observable: str):
    """
    This function calculates the expectation value of an observable
    given an input quantum state (expressed in the density matrix representation).

    Args:
        density_state (DensityMatrix):
            The input quantum state, given in the density matrix representation.

        observable (str):
            The observable for which this function calculates the expectation values of.

            For an n-qubit Pauli string P, where Pauli Pi acts on qubit qi, the order
            in the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit observable string 'YX', qubit q0 is measured
            in the X basis, while qubit q1 is measured in the Y basis.
    """
    
    observable_operator = Pauli(observable)
    observable_expectation_value = density_state.expectation_value(observable_operator)

    return observable_expectation_value