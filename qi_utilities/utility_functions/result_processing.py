from qiskit import QuantumCircuit

def return_raw_data(qc: QuantumCircuit,
                    result,
                    circuit_nr: int = None):

    bit_register_size = qc.num_clbits
    raw_data = result.get_memory(circuit_nr)
    for entry in range(len(raw_data)):
        additional_len = bit_register_size - len(raw_data[entry])
        for i in range(additional_len):
            raw_data[entry] = '0' + raw_data[entry]

    return raw_data

def get_raw_data_counts(qc: QuantumCircuit,
                        result,
                        circuit_nr: int = None):

    raw_data = return_raw_data(qc, result, circuit_nr)
    raw_data_counts = []

    bit_register_size = len(raw_data[0])
    measurement_shots = len(raw_data)
    for bit_index in range(bit_register_size):
        counter_0 = 0
        counter_1 = 0
        for meas_index in range(measurement_shots):
            if raw_data[meas_index][-bit_index-1] == '0':
                counter_0 += 1
            if raw_data[meas_index][-bit_index-1] == '1':
                counter_1 += 1
        raw_data_counts.append({'0': counter_0, '1': counter_1})
    
    return raw_data_counts

def get_raw_data_prob(qc: QuantumCircuit,
                      result,
                      circuit_nr: int = None):
    raw_data_counts = get_raw_data_counts(qc, result, circuit_nr)
    raw_data_prob = []

    measurement_shots = raw_data_counts[0]['0'] + raw_data_counts[0]['1']

    for entry in range(len(raw_data_counts)):
        prob_0 = raw_data_counts[entry]['0'] / measurement_shots
        prob_1 = raw_data_counts[entry]['1'] / measurement_shots

        raw_data_prob.append({'prob(0)': prob_0, 'prob(1)': prob_1})
    return raw_data_prob