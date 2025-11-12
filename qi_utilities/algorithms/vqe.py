from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.linalg import expm
import qiskit.quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate
from qi_utilities.utility_functions.circuit_modifiers import (initialization, apply_pre_measurement_rotations,
                                                              apply_readout_circuit)

def hva_block(quantum_circuit, Hamiltonian, var_parameters, parameter_idx):
    Hamiltonian = Hamiltonian[::-1]
    for entry in range(len(Hamiltonian)):
        pauli_string = Hamiltonian.to_list()[entry][0]

        string_to_list = list(pauli_string)
        target_qubits = []
        qubit_counter = 0
        for index in reversed(range(len(string_to_list))):
            if string_to_list[index] == 'I':
                pass
            else:
                target_qubits.append(qubit_counter)
            qubit_counter += 1

        unitary_label = 'HVA block,' + f' Pauli: {pauli_string}'.upper() + \
                        f'\nθ[{parameter_idx}]'
        unitary_gate = PauliEvolutionGate(operator = Hamiltonian[entry],
                                          time = var_parameters[parameter_idx],
                                          label = unitary_label)
        quantum_circuit.append(unitary_gate, target_qubits)
        parameter_idx += 1
    
    return parameter_idx

def construct_hva_circuit(nr_qubits,
                          initial_state,
                          hamiltonian_operator,
                          repetitions):

    bit_register_size = 0
    hamiltonian_terms = []
    for entry in range(len(hamiltonian_operator)):
        hamiltonian_terms.append(hamiltonian_operator.to_list()[entry][0])
        bit_register_size += len(hamiltonian_terms[entry])

    qc = QuantumCircuit(nr_qubits, bit_register_size)

    var_parameters = ParameterVector('θ', len(hamiltonian_terms)*repetitions) #ex. parameters[0] is the first parameter
    bit_register_idx = 0
    for observable in hamiltonian_terms:
        parameter_idx = 0
        qc = initialization(qc,
                            nr_qubits,
                            initial_state)
        for repetition in range(1, repetitions + 1):
            parameter_idx = hva_block(quantum_circuit=qc,
                                    Hamiltonian=hamiltonian_operator,
                                    var_parameters=var_parameters,
                                    parameter_idx=parameter_idx)
        apply_pre_measurement_rotations(qc, observable, [bit_register_idx, bit_register_idx+len(observable)-1])
        bit_register_idx += len(observable)
        qc.barrier()

    qc = apply_readout_circuit(qc, nr_qubits)

    return qc

def hardware_efficient_vqe(num_qubits, circuit_depth):
    parameters = ParameterVector('θ', num_qubits*(3*circuit_depth + 2)) #ex. parameters[0] is the first parameter
    vqe_circuit = QuantumCircuit(num_qubits, num_qubits) #Quantum Circuit
    for i in range(num_qubits):
        vqe_circuit.initialize(0,i)
        vqe_circuit.rx(parameters[2*i], i)
        vqe_circuit.rz(parameters[2*i+1], i)
    for depth in range(circuit_depth):
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if j == i:
                    break
                vqe_circuit.cz(i, j)
            vqe_circuit.rz(parameters[2*num_qubits + 3*(num_qubits*depth + i)], i)
            vqe_circuit.rx(parameters[2*num_qubits + 3*(num_qubits*depth + i) + 1], i)
            vqe_circuit.rz(parameters[2*num_qubits + 3*(num_qubits*depth + i) + 2], i)
        
    return vqe_circuit


# vqe_circuit = hardware_efficient_vqe(num_qubits, circuit_depth)
# initial_point = np.random.random(vqe_circuit.num_parameters)

# def energy_function(parameter_values):
#     ansatz_circuit_XX = make_ansatz(num_qubits, circuit_depth, parameter_values)
#     ansatz_circuit_YY = make_ansatz(num_qubits, circuit_depth, parameter_values)
#     ansatz_circuit_ZZ = make_ansatz(num_qubits, circuit_depth, parameter_values)
    
#     X1X0 = measure_observable(['I', 'X', 'X', 'I', 'I'], ansatz_circuit_XX, backend['backend_type'], experiment_backend, simulator_gate_set,
#                               simulator_coupling_map, simulator_noise_model, backend['experiment_shots'])
#     Y1Y0 = measure_observable(['I', 'Y', 'Y', 'I', 'I'], ansatz_circuit_YY, backend['backend_type'], experiment_backend, simulator_gate_set,
#                               simulator_coupling_map, simulator_noise_model, backend['experiment_shots'])
#     Z1Z0 = measure_observable(['I', 'Z', 'Z', 'I', 'I'], ansatz_circuit_ZZ, backend['backend_type'], experiment_backend, simulator_gate_set,
#                               simulator_coupling_map, simulator_noise_model, backend['experiment_shots'])

#     energy_value = J * (X1X0 + Y1Y0 + Z1Z0)
    
#     global iteration_number
#     iteration_number += 1
#     print(f'Optimization round: ', iteration_number)
    
#     return energy_value

# optimizer = SPSA(maxiter=spsa_iterations)
# result = optimizer.minimize(fun=energy_function, x0=initial_point)
# print(result.fun)