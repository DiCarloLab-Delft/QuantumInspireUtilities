"""
Utility functions for constructing and modifying specific parts
of an existing Qiskit QuantumCircuit object ('qc').

This module contains helper routines for state initialization,
measurement basis rotation, and readout circuit composition.

Authors: Marios Samiotis
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

def prepare_initial_state(qc: QuantumCircuit,
                          initial_state: str):
    """
    This function applies an initialization to an existing
    QuantumCircuit object. The initialization works only for either
    the state |0> or |1>.
    
    Args:
        qc (QuantumCircuit):
            The quantum circuit object.

        initial_state (str):
            A bitstring specifying the initial state. The order in the bitstring
            is 'qn-1,qn-2,...,q2,q1,q0'. The string can only contain '0' or '1'.
            e.g. for a quantum circuit with 3 qubits, initial_state = '100' will
            initialize qubits q0 and q1 in the state |0>, and qubit q2 in the state
            |1>.
    """
    
    nr_qubits = qc.num_qubits
    qc.barrier()
    for idx in range(nr_qubits):
        qc.reset(idx)
    qc.barrier()

    if len(initial_state) != nr_qubits:
        raise ValueError('Initial state must have same number of qubits defined.')
    for idx in range(len(initial_state)):
        if initial_state[idx] == '1':
            qc.x((nr_qubits-1) - idx)
    qc.barrier()

    return qc

def apply_pre_measurement_rotations(qc: QuantumCircuit,
                                    observable: str,
                                    bit_register: list = None):
    """
    This function applies all necessary pre-rotations prior to the measurement
    of an observable. It is meant to be applied on a QuantumCircuit object which
    does not contain any measurement blocks at its end.

    Args:
        qc (QuantumCircuit):
            The quantum circuit object.

        observable (str):
            The observable to be measured, expressed strictly in the Pauli basis,
            for which the pre-measurement rotations are being applied for.
            For an n-qubit Pauli string P, where Pauli Pi acts on qubit qi, the order
            in the string is 'Pn-1,Pn-2,...,P2,P1,P0'.
            e.g. for the two-qubit observable string 'YX', qubit q0 is measured
            in the X basis, while qubit q1 is measured in the Y basis.

        bit_register (list):
            A list specifying the bit register for which the measurements outcomes
            will be stored in.
    """
    
    nr_qubits = len(observable)
    for idx in range(nr_qubits):
        if observable[idx] == 'I':
            return qc
        
        elif observable[idx] == 'X':
            qc.ry(-np.pi/2,(nr_qubits-1)-idx)
        elif observable[idx] == 'Y':
            qc.rx(np.pi/2,(nr_qubits-1)-idx)
        elif observable[idx] == 'Z':
            pass

        if bit_register is not None:
            qc.measure((nr_qubits-1)-idx, bit_register[(nr_qubits-1)-idx])
        else:
            qc.measure((nr_qubits-1)-idx,(nr_qubits-1)-idx)

    return qc

def apply_readout_circuit(qc: QuantumCircuit,
                          qubit_list: list):
    """
    This function appends a QuantumCircuit object with a quantum circuit
    containing mid-circuit measurements which are used in post-processing
    for constructing the readout assignment matrix. The matrix is later on
    applied on the raw measurement outcomes in order to mitigate readout
    errors.

    Args:
        qc (QuantumCircuit):
            The quantum circuit object.

        qubit_list (list):
            An ordered list specifying the qubits on which we apply the
            readout circuit on.
            e.g. qubit_list = [0, 2] will apply the readout circuit on
            qubits q0 and q2.
    """

    nr_qubits = len(qubit_list)
    readout_circuit = QuantumCircuit(qc.num_qubits, qc.num_clbits + nr_qubits * 2**nr_qubits, name=qc.name)

    binary_list = []
    for binary_str_idx in range(2**nr_qubits):
        binary_list.append(np.binary_repr(binary_str_idx, nr_qubits))

    for binary_str_idx in range(len(binary_list)):

        reversed_binary_string = binary_list[binary_str_idx][::-1]

        for qubit_idx in qubit_list:
            readout_circuit.reset(qubit_idx)

        for idx in range(len(reversed_binary_string)):
            if reversed_binary_string[idx] == '1':
                readout_circuit.x(qubit_list[idx])
        readout_circuit.barrier()

        readout_circuit.measure(qubit_list,
                                list(np.arange(start=qc.num_clbits + binary_str_idx*nr_qubits,
                                               stop=qc.num_clbits + (binary_str_idx+1)*nr_qubits, step=1)))
        readout_circuit.barrier()

    qc.barrier()
    additional_bits = ClassicalRegister(nr_qubits * 2**nr_qubits)
    qc.add_bits(additional_bits)
    return readout_circuit.compose(qc, front=True)