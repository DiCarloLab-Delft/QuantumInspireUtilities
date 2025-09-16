from scipy.linalg import expm
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

def trotter_block(quantum_circuit, Hamiltonian, n_order, time):
    Hamiltonian = Hamiltonian[::-1]
    for entry in range(len(Hamiltonian)):
        pauli_string = Hamiltonian.to_list()[entry][0]

        string_to_list = list(pauli_string)
        target_qubits = []
        pauli_list = []
        qubit_counter = 0
        for index in reversed(range(len(string_to_list))):
            if string_to_list[index] == 'I':
                pass
            else:
                target_qubits.append(qubit_counter)
                pauli_list.append(string_to_list[index])
            qubit_counter += 1
        pauli_list.reverse()
        pauli_term = "".join(pauli_list)

        pauli_matrix = qi.Operator(qi.Pauli(pauli_term)).to_matrix()
        pauli_coefficient = Hamiltonian.to_list()[entry][1]
        pauli_exponentiate = expm(- 1j * pauli_coefficient * pauli_matrix * (time / n_order))
        unitary_gate = qi.Operator(pauli_exponentiate).to_instruction()
        unitary_gate.name = 'Trotter block,' + f' Pauli: {pauli_string}'.upper() + \
            f'\nn = {n_order},' + f' Time = {time*1e9:.2f} ns'

        quantum_circuit.append(unitary_gate, target_qubits)


def construct_trotterization_circuit(nr_qubits,
                                     initial_state,
                                     measured_observable,
                                     hamiltonian_operator,
                                     trotter_order,
                                     evolution_times,
                                     time_step,
                                     midcircuit_measurement = False):

    if midcircuit_measurement == True:
        qc = QuantumCircuit(nr_qubits, nr_qubits*len(evolution_times))
    else:
        qc = QuantumCircuit(nr_qubits, nr_qubits)

    for idx in range(nr_qubits):
        qc.reset(idx)
    qc.barrier()

    if len(initial_state) != nr_qubits:
        raise ValueError('Initial state must have same number of qubits defined.')
    for idx in range(len(initial_state)):
        if initial_state[idx] == '1':
            qc.x((nr_qubits-1) - idx)
    qc.barrier()

    for repetition in range(1, trotter_order+1):
        trotter_block(qc, hamiltonian_operator, trotter_order, evolution_times[time_step])
    qc.barrier()

    if midcircuit_measurement == True:
        apply_pre_measurement_rotations(qc, measured_observable, [nr_qubits*time_step, nr_qubits*time_step + 1])
    else:
        apply_pre_measurement_rotations(qc, measured_observable)
    qc.barrier()

    return qc