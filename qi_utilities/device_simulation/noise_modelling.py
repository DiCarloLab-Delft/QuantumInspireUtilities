from qiskit_aer import noise
from scipy.constants import Planck, Boltzmann
from qiskit.quantum_info import pauli_basis
import ast
import numpy as np

T1_FUDGE_FACTOR = 0.6
T2_FUDGE_FACTOR = 0.6

def relaxation_channel(temperature, qubit_freq, qubit_T1, time_interval):
    kraus_op = []
    decay_prob = 1 / (1 + np.exp( (-Planck * qubit_freq) / (Boltzmann * temperature) ))
    gamma_rate = 1 - np.exp( - time_interval / qubit_T1)
    kraus_op.append( [[np.sqrt(decay_prob), 0],
                      [0, np.sqrt(decay_prob * (1 - gamma_rate))]] )
    kraus_op.append( [[0, np.sqrt(decay_prob * gamma_rate)],
                      [0, 0]] )
    kraus_op.append( [[np.sqrt((1 - decay_prob)*(1 - gamma_rate)), 0],
                      [0, np.sqrt(1 - decay_prob)]] )
    kraus_op.append( [[0, 0],
                      [np.sqrt(gamma_rate * (1 - decay_prob)), 0]] )
    return kraus_op

def pure_dephasing_channel(qubit_T1, qubit_T2, time_interval):
    kraus_op = []
    qubit_Tphi = 1 / (1/qubit_T2 - 1/(2*qubit_T1))
    lambda_rate = 1 - np.exp(- time_interval / qubit_Tphi)
    kraus_op.append( [[1, 0],
                      [0, np.sqrt(1 - lambda_rate)]] )
    kraus_op.append( [[0, 0],
                      [0, np.sqrt(lambda_rate)]])
    return kraus_op

def depolarization_channel(num_qubits,
                           epsilon_cl):
    kraus_op = []
    pauli_basis_list = pauli_basis(num_qubits=num_qubits)
    
    p_RB_decay = 1 - (2**num_qubits * epsilon_cl)/(2**num_qubits - 1)
    lambda_rate = (2**(2*num_qubits)-1)/2**(2*num_qubits) * (1-p_RB_decay)
    kraus_coeff_0 = np.sqrt(1 - lambda_rate)
    kraus_coeff = np.sqrt(lambda_rate / (2**(2*num_qubits) - 1))
    
    kraus_op.append(kraus_coeff_0 * pauli_basis_list[0].to_matrix())
    for pauli_idx in range(1, len(pauli_basis_list)):
        kraus_op.append(kraus_coeff * pauli_basis_list[pauli_idx].to_matrix())
        
    return kraus_op


def create_noise_model(processor_specs: dict):
    
    n_g = 1.875
    n_CZ = 1.5
    
    qubit_list = list(processor_specs['Qubits'])
    CZ_list = list(processor_specs['CZ IRB errors'])
    
    simulator_noise_model = noise.NoiseModel()
    
    for qubit_idx in range(len(qubit_list)):
        
        decay_prob = 1 / (1 + np.exp( (-Planck * processor_specs['Qubits'][qubit_list[qubit_idx]]['Frequency [Hz]']) / (Boltzmann * processor_specs['Base temperature [K]']) ))
        relaxation_dephasing_delay = noise.thermal_relaxation_error(t1 = T1_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T1 [s]'] * 1e6,
                                                                    t2 = T2_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T2 [s]'] * 1e6,
                                                                    time = 1,
                                                                    excited_state_population = 1 - decay_prob)
        relaxation_dephasing_measure = noise.thermal_relaxation_error(t1 = T1_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T1 [s]'] * 1e6,
                                                                      t2 = T2_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T2 [s]'] * 1e6,
                                                                      time = processor_specs['Measurement duration [s]'] * 1e6,
                                                                      excited_state_population = 1 - decay_prob)
        
        epsilon_cl = 1 - (1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['RB error'])**n_g
        depolarizing_noise = noise.kraus_error(depolarization_channel(num_qubits = 1,
                                                                      epsilon_cl = epsilon_cl))
        
        readout_error = noise.ReadoutError([[1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p1given0'],
                                             processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p1given0']],
                                             [processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p0given1'],
                                              1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p0given1']]])
        
        simulator_noise_model.add_quantum_error(relaxation_dephasing_delay, ['delay'], [qubit_idx], warnings=False)
        simulator_noise_model.add_quantum_error(relaxation_dephasing_measure, ['measure'], [qubit_idx], warnings=False)
        simulator_noise_model.add_quantum_error(depolarizing_noise, ['id', 's', 'sdg', 't', 'tdg', 'x', 'rx', 'y', 'ry', 'z'], [qubit_idx], warnings=False)
        simulator_noise_model.add_readout_error(readout_error, [qubit_idx])
        
    for CZ_idx in range(len(CZ_list)):
        
        epsilon_cl = 1 - (1 - processor_specs['CZ IRB errors'][CZ_list[CZ_idx]])**n_CZ
        depolarizing_noise_CZ = noise.kraus_error(depolarization_channel(num_qubits = 2,
                                                                         epsilon_cl = epsilon_cl))
        
        simulator_noise_model.add_quantum_error(depolarizing_noise_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx]), warnings=False)
        # simulator_noise_model.add_quantum_error(depolarizing_noise_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx])[::-1], warnings=False) # should this be here??
        # test with simple Bell state with CZ in both directions

    return simulator_noise_model