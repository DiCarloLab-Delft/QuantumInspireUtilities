"""
Utility functions for applying pauli twirling to CZ-gates.

Authors: Jan Hemink
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, overload

import numpy as np
from qiskit.circuit import Barrier
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()

# pre-computed pauli twirls for CZ gate.
# Has structure: [[q0_before, q1_before, q0_after, q1_after], global_phase]
CZ_PAULI_TWIRL = [
    [[X, X, Y, Y], 0],
    [[X, Y, Y, X], np.pi],
    [[X, Z, X, I], 0],
    [[X, I, X, Z], 0],
    [[Y, X, X, Y], np.pi],
    [[Y, Y, X, X], 0],
    [[Y, Z, Y, I], 0],
    [[Y, I, Y, Z], 0],
    [[Z, X, I, X], 0],
    [[Z, Y, I, Y], 0],
    [[Z, Z, Z, Z], 0],
    [[Z, I, Z, I], 0],
    [[I, X, Z, X], 0],
    [[I, Y, Z, Y], 0],
    [[I, Z, I, Z], 0],
    [[I, I, I, I], 0],
]


@overload
def pauli_twirl_cz(
    qc: QuantumCircuit,
    num_circuits: int,
    seed: int | float | str | bytes | bytearray | None = None
) -> list[QuantumCircuit]: ...


@overload
def pauli_twirl_cz(
    qc: QuantumCircuit,
    num_circuits: None = None,
    seed: int | float | str | bytes | bytearray | None = None,
) -> QuantumCircuit: ...


def pauli_twirl_cz(
    qc: QuantumCircuit,
    num_circuits: int | None = None,
    seed: int | float | str | bytes | bytearray | None = None,
) -> QuantumCircuit | list[QuantumCircuit]:
    """
    Applies pauli twirling to CZ-gates in a QuantumCircuit.

    A randomly sampled 2-qubit pauli operator is applied before every CZ-gate in the
    circuit, with the corresponding 2-qubit pauli operator being applied after the CZ-gate
    such that the ideal action of the circuit does not change.

    As this function only applies the twirling to CZ-gates, it is important to make sure
    that the only 2-qubit gates the circuit contains are CZ-gates. This can be achieved
    by transpiling the circuit beforehand, as the CZ-gate is the only native two-qubit
    for the superconducting backends of QuantumInspire.

    Parameters
    ----------
        qc : QuantumCircuit
            The (transpiled) circuit containing CZ-gates.
        num_circuits : int or None, optional
            The number of twirled circuits that will be returned.
            Defaults to `None`, which will return a single circuit. If provided as an
            integer, a list of that many circuits will be returned.
        seed : int or float or string or bytes or bytearray or None, optional
            The seed to use for the random number generator, value will be passed to
            `random.seed(seed)`. Defaults to `None` , in which case `random.seed()` will
            use the current system time as the seed.

    Returns
    -------
    QuantumCircuit or list of QuantumCircuits
        The circuit with CZ-gates pauli twirled or list of twirled circuits
        (if `num_circuits` is provided).

    See Also
    --------
    `_twirl_cz_gates` for the actual implementation of twirling the CZ-gates.

    """
    random.seed(seed)
    if num_circuits is None:
        return _twirl_cz_gates(qc)

    list_twirled_circuits = []
    for _ in range(num_circuits):
        list_twirled_circuits.append(_twirl_cz_gates(qc))
    return list_twirled_circuits


def _twirl_cz_gates(
    qc: QuantumCircuit,
) -> QuantumCircuit:
    """
    Function that performs the actual pauli twirling of the CZ-gates.
    See `pauli_twirl_cz` for more information.
    """
    twirled_qc = qc.copy_empty_like()
    for instruction in qc:
        if instruction.name != "cz":  # if not a CZ-gate, just copy it to new circuit
            twirled_qc._data.append(instruction)
            continue

        qubits = instruction.qubits
        pauli_gates, phase = random.choice(CZ_PAULI_TWIRL)  # choose random pauli gates
        if phase != 0:
            twirled_qc.global_phase += phase
        # add gates to new circuit
        twirled_qc.append(pauli_gates[0], (qubits[0],))
        twirled_qc.append(pauli_gates[1], (qubits[1],))
        twirled_qc.append(Barrier(2), qubits)
        twirled_qc._data.append(instruction)
        twirled_qc.append(Barrier(2), qubits)
        twirled_qc.append(pauli_gates[2], (qubits[0],))
        twirled_qc.append(pauli_gates[3], (qubits[1],))

    return twirled_qc
