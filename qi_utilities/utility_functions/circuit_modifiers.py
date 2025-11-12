import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

def initialization(quantum_circuit: QuantumCircuit,
                   nr_qubits: int,
                   initial_state: str):
    
    quantum_circuit.barrier()
    for idx in range(nr_qubits):
        quantum_circuit.reset(idx)
    quantum_circuit.barrier()

    if len(initial_state) != nr_qubits:
        raise ValueError('Initial state must have same number of qubits defined.')
    for idx in range(len(initial_state)):
        if initial_state[idx] == '1':
            quantum_circuit.x((nr_qubits-1) - idx)
    quantum_circuit.barrier()

    return quantum_circuit

def apply_pre_measurement_rotations(qc: QuantumCircuit,
                                    observable: str,
                                    bit_register: list = None):
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
                          nr_qubits: int):

    readout_circuit = QuantumCircuit(nr_qubits, qc.num_clbits + 2**(nr_qubits+1), name=qc.name)

    binary_list = []
    for binary_str_idx in range(2**nr_qubits):
        binary_list.append(np.binary_repr(binary_str_idx, nr_qubits))

    for binary_str_idx in range(len(binary_list)):

        for qubit_idx in range(nr_qubits):
            readout_circuit.reset(qubit_idx)

        for idx in range(len(binary_list[binary_str_idx])):
            if binary_list[binary_str_idx][idx] == '1':
                readout_circuit.x((nr_qubits-1) - idx)
        readout_circuit.barrier()

        readout_circuit.measure(list(np.arange(stop=nr_qubits, step=1)),
                                list(np.arange(start=qc.num_clbits + binary_str_idx*nr_qubits,
                                               stop=qc.num_clbits + (binary_str_idx+1)*nr_qubits, step=1)))
        readout_circuit.barrier()

    additional_bits = ClassicalRegister(2**(nr_qubits+1))
    qc.add_bits(additional_bits)
    return readout_circuit.compose(qc, front=True)