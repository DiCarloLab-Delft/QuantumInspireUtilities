import numpy as np
from datetime import datetime
from qiskit import transpile
from qi_utilities.utility_functions.circuit_modifiers import apply_readout_circuit
from qi_utilities.utility_functions.raw_data_processing import get_multi_counts, get_multi_probs
from qi_utilities.utility_functions.readout_correction import (split_raw_shots, extract_ro_assignment_matrix,
                                                               get_ro_corrected_multi_probs, extract_ro_assignment_matrices)
from qi_utilities.utility_functions.data_handling import StoreProjectRecord

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class BaseMeasurement:
    """
    By default all measurement functions should have calibration
    points, meaning that every qc has additional circuits at the end
    for extracting the ro assignment matrix of each qubit
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 qc,
                 num_shots: int,
                 use_ro_cal_points: bool = True,
                 n_qubit_routine: int = 1,
                 qubit_groups: list[list] = None,
                 directory: str = None,
                 store_record: bool = True):
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              "Initializing measurement routine ...")

        if n_qubit_routine > 1:
            if qubit_groups is None:
                timestamp_now = datetime.now().replace(microsecond=0)
                raise ValueError(f"[{timestamp_now}] Since n_qubit_routine > 1, the qubit_groups need to be provided!")
        if qubit_groups is None:
            qubit_groups = []
            for qubit_idx in qubit_list:
                qubit_groups.append([qubit_idx])

        self.qc = qc    
        self.backend = backend
        self.qubit_list = qubit_list
        self.num_shots = num_shots

        for qubit_list in qubit_groups:
            if type(self.qc) == list:
                for qc_idx in range(len(self.qc)):
                    self.qc[qc_idx] = apply_readout_circuit(self.qc[qc_idx],
                                                            qubit_list)
            else:
                self.qc = apply_readout_circuit(self.qc,
                                                qubit_list)  
        result = self._create_and_execute_job(directory,
                                              store_record)
        self._extract_data(result,
                           use_ro_cal_points,
                           qubit_groups,
                           n_qubit_routine)
        if use_ro_cal_points == True:
            self.result_probs_per_n_qubits = self.ro_corrected_probs_per_n_qubits.copy()
            self.result_multi_probs_per_n_qubits = self.multi_ro_corrected_probs_per_n_qubits.copy()
        else:
            self.result_probs_per_n_qubits = self.raw_data_probs_per_n_qubits.copy()
            self.result_multi_probs_per_n_qubits = self.multi_raw_data_probs_per_n_qubits.copy()
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Experiment raw counts processed into probabilities.")

    def _create_and_execute_job(self,
                                directory: str = None,
                                store_record: bool = True):
        
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
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Uploaded schedule for execution on {self.backend.name} backend. Waiting for job results ...")
        result = job.result(timeout = 24 * 60 * 60) # a total of 24 hours
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Schedule execution complete, retrieved job results.")
        if store_record == True:
            timestamp_now = datetime.now().replace(microsecond=0)
            print(f"[{timestamp_now}] "
                  f"Storing locally the job record ...")
            self.record = StoreProjectRecord(job,
                                            directory,
                                            silent = True)
            timestamp_now = datetime.now().replace(microsecond=0)
            print(f"[{timestamp_now}] "
                  f"Job record successfully stored in local directory: {self.record.project_dir}")
        return result
        
    def _extract_data(self,
                      result,
                      use_ro_cal_points: bool = True,
                      qubit_groups: list[list] = None,
                      n_qubit_routine: int = 1,
                      check_for_multi_result: bool = True,
                      circuit_idx: int = None):
        
        if check_for_multi_result == True:
            self.multi_raw_data_probs_per_n_qubits = []
            self.multi_ro_corrected_probs_per_n_qubits = []
            if len(result.results) > 1:
                for circuit_idx in range(len(result.results)):
                    self._extract_data(result,
                                       use_ro_cal_points,
                                       qubit_groups,
                                       n_qubit_routine,
                                       check_for_multi_result=False,
                                       circuit_idx=circuit_idx)
                    self.multi_raw_data_probs_per_n_qubits.append(self.raw_data_probs_per_n_qubits)
                    if use_ro_cal_points == True:
                        self.multi_ro_corrected_probs_per_n_qubits.append(self.ro_corrected_probs_per_n_qubits)
        ro_register_length = 0
        for qubit_list in qubit_groups:
            ro_register_length += len(qubit_list)*2**len(qubit_list)
        raw_data_shots, ro_mitigation_shots = split_raw_shots(result,
                                                              self.qubit_list,
                                                              ro_register_length,
                                                              circuit_idx)
        
        ro_assignment_matrices = extract_ro_assignment_matrices(ro_mitigation_shots, qubit_groups)
        raw_data_counts = get_multi_counts(raw_data_shots, len(self.qubit_list))
        raw_data_probs = get_multi_probs(raw_data_counts)

        self.ro_corrected_probs_per_n_qubits = []
        self.raw_data_probs_per_n_qubits = self._collect_data_per_n_qubits(raw_data_probs,
                                                                           n_qubit_routine)
        if use_ro_cal_points == True:
            for qubit_list_idx in range(len(qubit_groups)):
                qubit_list = qubit_groups[qubit_list_idx]
                self.ro_corrected_probs_per_n_qubits.append(get_ro_corrected_multi_probs(self.raw_data_probs_per_n_qubits[qubit_list_idx],
                                                                                            ro_assignment_matrices[f"{qubit_list}"],
                                                                                            qubit_list))
    
    def _collect_data_per_n_qubits(self,
                                   ro_corrected_probs,
                                   n_qubit_routine: int = 1):
        
        binary_list = []
        for binary_str_idx in range(2**n_qubit_routine):
            binary_list.append(np.binary_repr(binary_str_idx, n_qubit_routine))
        
        ro_corrected_probs_per_qubit = []
        for qubit_idx in range(0, len(self.qubit_list), n_qubit_routine):
            ro_corrected_probs_qubit = []
            for measurement_idx in range(len(ro_corrected_probs)):
                meas_result = {bitstring:0 for bitstring in binary_list}
                for bitstring in ro_corrected_probs[measurement_idx].keys():
                    reversed_bitstring = bitstring[::-1]
                    for dict_bitstring in meas_result.keys():
                        if reversed_bitstring[qubit_idx:qubit_idx+n_qubit_routine] == dict_bitstring[::-1]:
                            meas_result[dict_bitstring] += ro_corrected_probs[measurement_idx][bitstring]
                            break
                ro_corrected_probs_qubit.append(meas_result)
            ro_corrected_probs_per_qubit.append(ro_corrected_probs_qubit)

        return ro_corrected_probs_per_qubit