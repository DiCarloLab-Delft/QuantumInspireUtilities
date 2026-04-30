import numpy as np
from qiskit import transpile
from qi_utilities.utility_functions.circuit_modifiers import apply_readout_circuit
from qi_utilities.utility_functions.raw_data_processing import get_multi_counts, get_multi_probs
from qi_utilities.utility_functions.readout_correction import (split_raw_shots, extract_ro_assignment_matrix,
                                                               get_ro_corrected_multi_probs)
from qi_utilities.utility_functions.data_handling import StoreProjectRecord

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class BaseMeasurement:
    """
    Meant for single-qubit experiments, which can run in parallel
    for multiple qubits.

    By default all measurement functions should have calibration
    points, meaning that every qc has additional circuits at the end
    for extracting the ro assignment matrix of each qubit
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 qc,
                 num_shots: int,
                 directory: str = None):

        self.backend = backend
        self.qubit_list = qubit_list
        self.num_shots = num_shots

        self.qc = apply_readout_circuit(qc,
                                        self.qubit_list)
        result = self._create_and_execute_job(directory)
        ro_corrected_probs = self._extract_data(result)
        self.ro_corrected_probs_per_qubit = self._collect_data_per_qubit(ro_corrected_probs)

    def _create_and_execute_job(self,
                                directory: str = None):
        
        tuna_backends_basis_gates = ['id', 's', 'sdg', 't', 'tdg', 'x', 'rx', 'y', 'ry', 'z', 'rz', 'cz', 'delay', 'reset']
        qc_transpiled = transpile(self.qc,
                                  self.backend,
                                  layout_method = "trivial",
                                  routing_method = "none",
                                  basis_gates=tuna_backends_basis_gates,
                                  optimization_level=0) # specific (and necessary) for the benchmark routines

        job = self.backend.run(qc_transpiled,
                          shots=self.num_shots,
                          memory = True) # NOTE: memory is set to True in order to return raw data!
        result = job.result(timeout = 600)
        self.record = StoreProjectRecord(job,
                                         directory,
                                         silent = True)
        return result
        
    def _extract_data(self,
                      result):

        raw_data_shots, ro_mitigation_shots = split_raw_shots(result, self.qubit_list)
        ro_assignment_matrix = extract_ro_assignment_matrix(ro_mitigation_shots, self.qubit_list)

        raw_data_counts = get_multi_counts(raw_data_shots, len(self.qubit_list))
        raw_data_probs = get_multi_probs(raw_data_counts)
        ro_corrected_probs = get_ro_corrected_multi_probs(raw_data_probs, ro_assignment_matrix, self.qubit_list)
        
        return ro_corrected_probs
    
    def _collect_data_per_qubit(self,
                                ro_corrected_probs):
        
        ro_corrected_probs_per_qubit = []
        for qubit_idx in range(len(self.qubit_list)):
            ro_corrected_probs_qubit = []
            for measurement_idx in range(len(ro_corrected_probs)):
                bit_0_prob = 0
                bit_1_prob = 0
                for bitstring in ro_corrected_probs[measurement_idx].keys():
                    if bitstring[-1 - qubit_idx] == '0':
                        bit_0_prob += ro_corrected_probs[measurement_idx][bitstring]
                    else:
                        bit_1_prob += ro_corrected_probs[measurement_idx][bitstring]
                meas_result = {
                    '0': bit_0_prob,
                    '1': bit_1_prob
                }
                ro_corrected_probs_qubit.append(meas_result)
            ro_corrected_probs_per_qubit.append(ro_corrected_probs_qubit)

        return ro_corrected_probs_per_qubit