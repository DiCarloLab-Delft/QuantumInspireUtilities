"""
Module implementing the generic base class "BaseMeasurement", which handles
the execution of the quantum circuit(s) of every characterization experiment
in measurement_classes.py, the storing of the project record, and the processing
of the raw data shots into (readout-error-mitigated) probabilities.

Authors: Marios Samiotis
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from qiskit import transpile
from qiskit_quantuminspire.qi_backend import QIBackend
from qi_utilities.device_simulation.simulators import NoisySimulator
from qi_utilities.utility_functions.circuit_modifiers import apply_readout_circuit
from qi_utilities.utility_functions.raw_data_processing import get_multi_counts, get_multi_probs
from qi_utilities.utility_functions.readout_correction import (split_raw_shots, get_ro_corrected_multi_probs,
                                                               extract_ro_assignment_matrices)
from qi_utilities.utility_functions.data_handling import StoreProjectRecord

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class BaseMeasurement:
    """
    This class is the generic parent class of all measurement classes. It is
    responsible for executing the experiment quantum circuit(s) on the backend,
    storing the project record locally, and processing the raw data shots into
    probabilities per measurement block.

    By default all measurement functions should have calibration
    points, meaning that every qc has additional circuits at the end
    for extracting the ro assignment matrix of each qubit
    (see apply_readout_circuit in circuit_modifiers.py).

    The final processed data are stored in the attribute
    'result_probs_per_n_qubits', a list containing, for each qubit
    (or qubit group, for n_qubit_routine > 1), a list of dictionaries
    with the measurement probabilities of each measurement block.
    e.g. for qubit_list = [0, 2] and three measurement blocks,
    result_probs_per_n_qubits = [[{'0': 0.9, '1': 0.1}, {...}, {...}],   # Q0
                                 [{'0': 0.8, '1': 0.2}, {...}, {...}]]   # Q2
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
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            qc (QuantumCircuit | list[QuantumCircuit]):
                The experiment quantum circuit(s), without the readout
                calibration circuits, which are appended here.

            num_shots (int):
                The number of shots for which the experiment is executed.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            n_qubit_routine (int):
                The number of qubits on which the measurement routine acts
                jointly, e.g. 1 for single-qubit and 2 for two-qubit experiments.

            qubit_groups (list(list)):
                An ordered list of lists (of the kind of qubit_list, see definition of
                extract_ro_assignment_matrix in readout_correction.py) containing the different integer
                groupings of qubits.
                e.g. qubit_groups = [[0, 2], [1, 3]].
                A separate readout assignment matrix is extracted for each group.
                It must be provided for n_qubit_routine > 1, otherwise it defaults to
                [[qubit_idx] for qubit_idx in qubit_list].

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

            store_record (bool):
                If True, the project record is stored locally.
        """

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
        self.backend_name = backend.name
        if type(backend) == QIBackend:
            self.backend_type = 'processor'
        elif type(backend) == NoisySimulator:
            self.backend_type = 'simulator'
        else:
            self.backend_type = None
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
        """
        This instance method transpiles the quantum circuit(s) to the native
        gate set of the Tuna backends, without any qubit re-mapping, routing or
        optimization, so that the experiment is executed exactly as defined.
        It then executes the job, waits for the result and stores the project
        record locally (see StoreProjectRecord in data_handling.py).

        Args:
            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

            store_record (bool):
                If True, the project record is stored locally.
        """

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
        """
        This instance method splits the raw data shots from the readout
        calibration shots, extracts the readout assignment matrices, and
        computes the raw (and readout-error-mitigated) probabilities per
        measurement block, grouped per n_qubit_routine qubits.

        For a job containing multiple quantum circuits, the probabilities of
        each circuit are additionally stored in the attributes
        'multi_raw_data_probs_per_n_qubits' and 'multi_ro_corrected_probs_per_n_qubits'.

        Args:
            result (Result):
                The result of a job (project), as returned from
                result = job.result()

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            qubit_groups (list(list)):
                An ordered list of lists (of the kind of qubit_list, see definition of
                extract_ro_assignment_matrix in readout_correction.py) containing the different integer
                groupings of qubits.
                e.g. qubit_groups = [[0, 2], [1, 3]].
                A separate readout assignment matrix is extracted for each group.

            n_qubit_routine (int):
                The number of qubits on which the measurement routine acts
                jointly.

            check_for_multi_result (bool):
                If True, checks whether the job contains multiple quantum circuits
                and processes each one of them separately.

            circuit_idx (int):
                The circuit number within a job, since a job can contain
                multiple quantum circuits.
                Defaults to None for a job with a single quantum circuit.
        """

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
        """
        This instance method marginalizes the multi-qubit probabilities of each
        measurement block onto consecutive groups of n_qubit_routine qubits,
        following the order of the qubit_list.
        e.g. for qubit_list = [0, 2, 1, 3] and n_qubit_routine = 2, the
        probabilities are marginalized onto the groups [0, 2] and [1, 3].

        Args:
            ro_corrected_probs (list):
                A list of dictionaries each containing the measurement probabilities
                of a certain measurement block.
                e.g. ro_corrected_probs = [{'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25},
                                           ...]

            n_qubit_routine (int):
                The number of qubits in each group.
        """

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
    
    def _load_probs_per_n_qubits(self,
                                 job_timestamp: str,
                                 directory: str = None):
        """
        This instance method searches the project directory for the JSON data
        file of a previously executed measurement, and loads its processed data
        and metadata into the measurement class attributes. It is used by the
        load_data class methods of the measurement classes.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        if directory is not None:
            project_dir = Path(directory)
        else:
            project_dir = Path.home() / "Documents" / "QuantumInspireProjects"

        json_file_path = None
        pattern = f"*_data_{job_timestamp}*.json"
        for path in project_dir.rglob(pattern):
            if path.is_file():
                json_file_path = path
                break
        if json_file_path is None:
            raise ValueError(f"No JSON file found for job timestamp: {job_timestamp}")
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)

        self.project_name = json_data['Experiment name']
        self.project_dir = json_file_path.parent
        self.date_timestamp = job_timestamp.split('_')[0]
        self.job_timestamp = job_timestamp.split('_')[1]
        self.backend_name = json_data["Backend name"]
        self.backend_type = json_data["Backend type"]
        self.num_shots = json_data["Number of shots"]
        self.result_probs_per_n_qubits = [json_data["Processed data"][key] for key in json_data["Processed data"].keys()]
        return json_data # return the whole data anyways