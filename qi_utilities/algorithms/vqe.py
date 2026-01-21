from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.linalg import expm
import qiskit.quantum_info as qi
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qi_utilities.utility_functions.circuit_modifiers import (prepare_initial_state, apply_pre_measurement_rotations,
                                                              apply_readout_circuit)
from datetime import datetime

def make_ansatz(vqe_circuit, parameter_values):
    ansatz_circuit = vqe_circuit
    ansatz_circuit = vqe_circuit.assign_parameters(parameter_values)
    return ansatz_circuit

def hva_block(qc: QuantumCircuit,
              hamiltonian: SparsePauliOp,
              repetitions: int,
              var_parameters,
              parameter_idx):
    hamiltonian = hamiltonian[::-1]
    for entry in range(len(hamiltonian)):
        pauli_string = hamiltonian.to_list()[entry][0]

        string_to_list = list(pauli_string)
        target_qubits = []
        qubit_counter = 0
        for index in reversed(range(len(string_to_list))):
            if string_to_list[index] == 'I':
                pass
            else:
                target_qubits.append(qubit_counter)
            qubit_counter += 1

        unitary_label = f'HVA block, rep: {repetitions}' + f'\nPauli: {pauli_string}'
        unitary_gate = PauliEvolutionGate(operator = hamiltonian[entry],
                                          time = var_parameters[parameter_idx],
                                          label = unitary_label)
        
        qc.append(unitary_gate, target_qubits)
        parameter_idx += 1
    
    return parameter_idx

def construct_hva_circuit(initial_state: str,
                          hamiltonian: SparsePauliOp,
                          repetitions: int):

    nr_qubits = len(initial_state)

    bit_register_size = 0
    hamiltonian_terms = []
    for entry in range(len(hamiltonian)):
        hamiltonian_terms.append(hamiltonian.to_list()[entry][0])
        bit_register_size += len(hamiltonian_terms[entry])

    qc = QuantumCircuit(nr_qubits,
                        bit_register_size,
                        name=f"HVA_{nr_qubits}_Qubits")

    var_parameters = ParameterVector('θ', len(hamiltonian_terms)*repetitions) #ex. parameters[0] is the first parameter
    bit_register_idx = 0
    for observable in hamiltonian_terms:
        parameter_idx = 0
        qc = prepare_initial_state(qc,
                            initial_state)
        for repetition_idx in range(1, repetitions + 1):
            parameter_idx = hva_block(qc=qc,
                                      hamiltonian=hamiltonian,
                                      repetitions=repetition_idx,
                                      var_parameters=var_parameters,
                                      parameter_idx=parameter_idx)
        apply_pre_measurement_rotations(qc, observable, [bit_register_idx, bit_register_idx+len(observable)-1])
        bit_register_idx += len(observable)
        qc.barrier()

    return qc

def hardware_efficient_vqe(num_qubits,
                           repetitions):
    parameters = ParameterVector('θ', num_qubits*(3*repetitions + 2)) #ex. parameters[0] is the first parameter
    vqe_circuit = QuantumCircuit(num_qubits, num_qubits) #Quantum Circuit
    for i in range(num_qubits):
        vqe_circuit.initialize(0,i)
        vqe_circuit.rx(parameters[2*i], i)
        vqe_circuit.rz(parameters[2*i+1], i)
    for repetition in range(repetitions):
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if j == i:
                    break
                vqe_circuit.cz(i, j)
            vqe_circuit.rz(parameters[2*num_qubits + 3*(num_qubits*repetition + i)], i)
            vqe_circuit.rx(parameters[2*num_qubits + 3*(num_qubits*repetition + i) + 1], i)
            vqe_circuit.rz(parameters[2*num_qubits + 3*(num_qubits*repetition + i) + 2], i)
        
    return vqe_circuit