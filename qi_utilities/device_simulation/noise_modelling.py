"""
Functions for creating noise models compatible with the
Qiskit Aer Simulator.

Authors: Marios Samiotis
"""

from qiskit_aer import noise
from scipy.constants import Planck, Boltzmann
import ast
import numpy as np

T1_FUDGE_FACTOR = 1.3
T2_FUDGE_FACTOR = 1.3

def depolarization_param(num_qubits,
                         epsilon_cl):
    
    p_RB_decay = 1 - (2**num_qubits * epsilon_cl)/(2**num_qubits - 1)
    lambda_rate = (2**(2*num_qubits)-1)/2**(2*num_qubits) * (1-p_RB_decay)
    return lambda_rate

def create_noise_model(processor_specs: dict):
    
    n_g = 1.875
    n_CZ = 1.5
    
    qubit_list = list(processor_specs['Qubits'])
    CZ_list = list(processor_specs['CZ IRB errors'])
    
    simulator_noise_model = noise.NoiseModel()
    
    for qubit_idx in range(len(qubit_list)):
        
        decay_prob = 1 / (1 + np.exp( (-Planck * processor_specs['Qubits'][qubit_list[qubit_idx]]['Frequency [Hz]']) / (Boltzmann * processor_specs['Base temperature [K]']) ))
        relaxation_dephasing_delay = noise.thermal_relaxation_error(t1 = T1_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T1 [s]'],
                                                                    t2 = processor_specs['Qubits'][qubit_list[qubit_idx]]['T2 [s]'],
                                                                    time = processor_specs['Delay duration [s]'],
                                                                    excited_state_population = 1 - decay_prob)
        relaxation_dephasing_measure = noise.thermal_relaxation_error(t1 = T1_FUDGE_FACTOR * processor_specs['Qubits'][qubit_list[qubit_idx]]['T1 [s]'],
                                                                      t2 = processor_specs['Qubits'][qubit_list[qubit_idx]]['T2 [s]'],
                                                                      time = processor_specs['Measurement duration [s]'],
                                                                      excited_state_population = 1 - decay_prob)
        
        epsilon_cl = 1 - (1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['RB error'])**n_g
        lambda_param = depolarization_param(num_qubits=1, epsilon_cl=epsilon_cl)
        depolarizing_error = noise.depolarizing_error(param=lambda_param, num_qubits=1)
        
        readout_error = noise.ReadoutError([[1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p1given0'],
                                             processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p1given0']],
                                             [processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p0given1'],
                                              1 - processor_specs['Qubits'][qubit_list[qubit_idx]]['SSRO']['p0given1']]])
        
        simulator_noise_model.add_quantum_error(relaxation_dephasing_delay, ['delay'], [qubit_idx], warnings=False)
        simulator_noise_model.add_quantum_error(relaxation_dephasing_measure, ['measure'], [qubit_idx], warnings=False)
        simulator_noise_model.add_quantum_error(depolarizing_error, ['id', 's', 'sdg', 't', 'tdg', 'x', 'rx', 'y', 'ry', 'z'], [qubit_idx], warnings=False)
        simulator_noise_model.add_readout_error(readout_error, [qubit_idx])
        
    for CZ_idx in range(len(CZ_list)):
        
        epsilon_cl = 1 - (1 - processor_specs['CZ IRB errors'][CZ_list[CZ_idx]])**n_CZ
        lambda_param = depolarization_param(num_qubits=2, epsilon_cl=epsilon_cl)
        depolarizing_error_CZ = noise.depolarizing_error(param=lambda_param, num_qubits=2)
        
        simulator_noise_model.add_quantum_error(depolarizing_error_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx]), warnings=False)
        simulator_noise_model.add_quantum_error(depolarizing_error_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx])[::-1], warnings=False)

    return simulator_noise_model