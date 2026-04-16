"""
Functions for creating noise models compatible with the
Qiskit Aer Simulator.

Authors: Marios Samiotis
"""

import ast
import numpy as np
from qiskit_aer import noise
from scipy.constants import Planck, Boltzmann

T1_FUDGE_FACTOR = 1.3
T2_FUDGE_FACTOR = 1.3

def depolarization_param(num_qubits: int,
                         epsilon_cl: float):
    """
    This function calculates the lambda parameter of an n-qubit depolarizing
    error channel.

    Args:
        num_qubits (int):
            The number of qubits for which the lambda parameter is estimated for.

        epsilon_cl (float):
            The error rate per Clifford, as that is extracted from a randomized
            benchmarking experiment.
    """
    p_RB_decay = 1 - (2**num_qubits * epsilon_cl)/(2**num_qubits - 1)
    lambda_rate = (2**(2*num_qubits)-1)/2**(2*num_qubits) * (1-p_RB_decay)
    return lambda_rate

def create_noise_model(processor_specs: dict,
                       noise_applied: dict = {
                             'delay_T1_T2': True,
                             'sq_depolarization': True,
                             'readout_T1_T2': True,
                             'readout_assignment': True,
                             'CZ_depolarization': True
                        }):
    """
    This function instantiates a Qiskit NoiseModel from a given processor
    specs dictionary. The noise model includes:

    * Single-qubit gate errors: single-qubit depolarizing channel.
    * Two-qubit gate errors: two-qubit depolarizing channel.
    * Single-qubit idling errors: relaxation + pure dephasing channels.
    * Readout errors: relaxation + pure dephasing channels for the duration
                      of the readout operation, as well as readout assignment
                      errors.

    The noise modelling does not capture leakage or crosstalk effects.

    Args:
        processor_specs (dict):
            The processor specs to be used for creating the noise model,
            as those are stored within the backend_parameters.json file.
    """
    
    # For the numbers n_g and n_CZ defined below see M. A. Rol PhD thesis
    # "Control for Programmable Superconducting Quantum Systems", Ch. A.1 pp. 165-166,
    # DOI: https://doi.org/10.4233/uuid:0a2ba212-f6bf-4c64-8f3d-b707f1e44953
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
        
        if noise_applied['delay_T1_T2'] == True:
            simulator_noise_model.add_quantum_error(relaxation_dephasing_delay, ['delay'], [qubit_idx], warnings=False)
        if noise_applied['sq_depolarization'] == True:
            simulator_noise_model.add_quantum_error(depolarizing_error, ['id', 's', 'sdg', 't', 'tdg', 'x', 'rx', 'y', 'ry', 'z'], [qubit_idx], warnings=False)
        if noise_applied['readout_T1_T2'] == True:
            simulator_noise_model.add_quantum_error(relaxation_dephasing_measure, ['measure'], [qubit_idx], warnings=False)
        if noise_applied['readout_assignment'] == True:
            simulator_noise_model.add_readout_error(readout_error, [qubit_idx])
        
    for CZ_idx in range(len(CZ_list)):
        
        epsilon_cl = 1 - (1 - processor_specs['CZ IRB errors'][CZ_list[CZ_idx]])**n_CZ
        lambda_param = depolarization_param(num_qubits=2, epsilon_cl=epsilon_cl)
        depolarizing_error_CZ = noise.depolarizing_error(param=lambda_param, num_qubits=2)
        
        if noise_applied['CZ_depolarization'] == True:
            simulator_noise_model.add_quantum_error(depolarizing_error_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx]), warnings=False)
            simulator_noise_model.add_quantum_error(depolarizing_error_CZ, ['cz'], ast.literal_eval(CZ_list[CZ_idx])[::-1], warnings=False)

    return simulator_noise_model