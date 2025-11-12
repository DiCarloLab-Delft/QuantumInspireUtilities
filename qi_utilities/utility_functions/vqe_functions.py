def make_ansatz(vqe_circuit, parameter_values):
    ansatz_circuit = vqe_circuit
    ansatz_circuit = vqe_circuit.assign_parameters(parameter_values)
    return ansatz_circuit