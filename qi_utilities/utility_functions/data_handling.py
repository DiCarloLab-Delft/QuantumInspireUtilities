"""
Utility classes for storing and retrieving a job (project)
data and metadata in a structured manner in the local user directory.

Authors: Marios Samiotis
"""

import numpy as np
import json
import h5py
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageFilter
from qiskit import qasm3
from qiskit_quantuminspire.cqasm import dumps
from qiskit_quantuminspire.qi_jobs import QIJob

class StoreProjectRecord:
    """
    This class is responsible for storing a job (project) record
    within the user's local 'Documents' directory.

    'Record' encapsulates both 'data', such as the measurement counts
    or even raw data shots, but also the 'metadata', such as information
    regarding the backend, the quantum circuits executed, etc.

    It works with both the Tuna backends but also the 'QX emulator' backend.
    """

    def __init__(self,
                 job: QIJob):
        """
        Args:
            job (QIJob):
                The user already-submitted job object, more correctly referred to as
                'project'. A project can contain multiple jobs, but for simplification,
                and also for legacy reasons, we keep referring to it as the 'job'.
        """

        self.create_project_directory(job)
        self.obtain_backend_metadata(job)
        self.store_project_json()
        for job_idx in range(len(job.circuits_run_data)):
            self.create_job_directory(job, job_idx)
            self.store_circuit_metadata(job, job_idx)
            self.store_job_result(job, job_idx)
            if self.raw_data_memory == True:
                self.store_raw_data(job, job_idx)

        return print(f"Successfully stored project record in the following directory:\n{str(self.project_dir)}\n")

    def create_project_directory(self,
                                 job: QIJob):
        """
        This instance method creates a new project folder within the local user
        Documents / QuantumInspireProjects directory. If this directory does not
        already exist, it will be created automatically.

        Args:
            job (QIJob):
                The user already-submitted job (project) object.
        """

        timestamp_utc = job.circuits_run_data[0].results.created_on # actually when the job finished, not when created
        timestamp = timestamp_utc.astimezone()
        self.date_timestamp = timestamp.strftime("%Y%m%d")
        self.job_0_timestamp = timestamp.strftime("%H%M%S")

        self.project_name = job.program_name
        self.project_dir = (
            Path.home() / "Documents" / "QuantumInspireProjects" / self.date_timestamp
            / f"{self.job_0_timestamp}_{self.project_name}"
        )
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def obtain_backend_metadata(self,
                                job: QIJob):
        """
        This instance method retrieves and then stores all relevant metadata
        for the backend which was used to run the job(s) on.

        Args:
            job (QIJob):
                The user already-submitted job (project) object.
        """

        self.backend_name = job.backend().name
        self.backend_nr_qubits = job.backend().num_qubits
        self.backend_operations = []
        for entry in range(len(job.backend().operations)):
            self.backend_operations.append(str(job.backend().operations[entry]))
        self.backend_max_shots = job.backend().max_shots

        try: # since the user may have used an emulator
            figure = job.backend().coupling_map.draw()
            image = figure.resize((800, 800))
            sharpened = image.filter(ImageFilter.SHARPEN)
            image_array = np.array(sharpened)

            file_path = (
                Path(self.project_dir)
                / f"backend_coupling_map_{self.date_timestamp}_{self.job_0_timestamp}.png"
            )

            plt.clf()
            plt.imshow(image_array)
            plt.title(f'\n{self.date_timestamp}_{self.job_0_timestamp}\n{job.backend().name} coupling map\n', fontsize=18)
            plt.axis('off')
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            exception_message = str(e)
            warnings.warn(f"\nFailed storing backend {self.backend_name} coupling map figure.\nError message: {exception_message}\n",
                          UserWarning)

    def store_project_json(self):
        """
        This instance method stores job (project) related metadata within
        the project directory in a JSON format.
        """
        
        general_dict = {}
        general_dict['Project name'] = self.project_name
        general_dict['Project timestamp'] = f"{self.date_timestamp}_{self.job_0_timestamp}"
        general_dict['Backend name'] = self.backend_name
        general_dict['Backend number of qubits'] = self.backend_nr_qubits
        general_dict['Backend operations set'] = self.backend_operations
        general_dict['Backend maximum allowed shots'] = self.backend_max_shots

        file_path = (
            Path(self.project_dir)
            / f"project_metadata_{self.date_timestamp}_{self.job_0_timestamp}.json"
        )
        with open(file_path, 'w') as file:
            json.dump(general_dict, file, indent=3)

    def create_job_directory(self,
                             job: QIJob,
                             job_idx: int):
        """
        This instance method creates new directories for each job contained
        within the project.

        Args:
            job (QIJob):
                The user already-submitted job (project) object.

            job_idx (int):
                The job index for all jobs contained within the project.
                While a project may contain a certain number of jobs, e.g. N,
                it is generally true that the execution of all these jobs
                in the Quantum Inspire platform is not sequential with respect
                to the order with which those jobs were created.
                Therefore, job_idx is being utilized for clarity when storing
                the data, so that it follows the sequence with which the jobs were
                created.
        """
        
        timestamp_utc = job.circuits_run_data[job_idx].results.created_on # actually when the job finished, not when created
        timestamp = timestamp_utc.astimezone()
        self.date_timestamp = timestamp.strftime("%Y%m%d")
        self.job_timestamp = timestamp.strftime("%H%M%S")
        self.job_id = job.circuits_run_data[job_idx].results.job_id

        self.job_dir = (
            Path.home()
            / "Documents" / "QuantumInspireProjects" / self.date_timestamp
            / f"{self.job_0_timestamp}_{self.project_name}"
            / f"job_idx_{job_idx}__job_id_{self.job_id}"
        )
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def store_job_result(self,
                         job: QIJob,
                         job_idx: int):
        """
        This instance method stores the job results (with the exception of
        the raw data), i.e. relevant job metadata and the result 'counts'.

        Args:
            job (QIJob):
                The user already-submitted job (project) object.

            job_idx (int):
                The job index for all jobs contained within the project.
                While a project may contain a certain number of jobs, e.g. N,
                it is generally true that the execution of all these jobs
                in the Quantum Inspire platform is not sequential with respect
                to the order with which those jobs were created.
                Therefore, job_idx is being utilized for clarity when storing
                the data, so that it follows the sequence with which the jobs were
                created.
        """
        
        self.result_id = job.circuits_run_data[job_idx].results.id
        self.execution_time_in_seconds = job.circuits_run_data[job_idx].results.execution_time_in_seconds
        self.shots_requested = job.circuits_run_data[job_idx].results.shots_requested
        self.shots_done = job.circuits_run_data[job_idx].results.shots_done
        if job.circuits_run_data[job_idx].results.raw_data == None:
            self.raw_data_memory = False
        else:
            self.raw_data_memory = True

        self.counts = job.circuits_run_data[job_idx].results.results

        job_result_dict = {}
        job_result_dict['Job timestamp'] = f"{self.date_timestamp}_{self.job_timestamp}"
        job_result_dict['Job ID'] = self.job_id
        job_result_dict['Result ID'] = self.result_id
        job_result_dict['Circuit name'] = self.circuit_name
        job_result_dict['Number of qubits specified'] = self.num_qubits
        job_result_dict['Number of classical bits specified'] = self.num_clbits
        job_result_dict['Circuit depth'] = self.circuit_depth
        job_result_dict['Execution time in seconds'] = self.execution_time_in_seconds
        job_result_dict['Shots requested'] = self.shots_requested
        job_result_dict['Shots done'] = self.shots_done
        job_result_dict['Raw data memory'] = self.raw_data_memory
        job_result_dict['Counts'] = self.counts
        file_path = (
            Path(self.job_dir)
            / f"job_result_{self.date_timestamp}_{self.job_timestamp}.json"
        )
        with open(file_path, 'w') as file:
            json.dump(job_result_dict, file, indent=3)

    def store_circuit_metadata(self,
                               job: QIJob,
                               job_idx: int):
        """
        This instance method stores the job circuit metadata, i.e. bookkeeping
        records, cQASM_v3 and OpenQASM3 program files, as well as the circuit
        figure.

        Args:
            job (QIJob):
                The user already-submitted job (project) object.

            job_idx (int):
                The job index for all jobs contained within the project.
                While a project may contain a certain number of jobs, e.g. N,
                it is generally true that the execution of all these jobs
                in the Quantum Inspire platform is not sequential with respect
                to the order with which those jobs were created.
                Therefore, job_idx is being utilized for clarity when storing
                the data, so that it follows the sequence with which the jobs were
                created.
        """

        self.qc = job.circuits_run_data[job_idx].circuit
        self.circuit_name = self.qc.name
        self.num_qubits = self.qc.to_instruction().num_qubits
        self.num_clbits = self.qc.to_instruction().num_clbits
        self.circuit_depth = self.qc.depth()

        qasm3_program = qasm3.dumps(self.qc)
        cqasm_v3_program = dumps(self.qc)
        qasm3_program_path = (
            Path(self.job_dir)
            / f"qasm3_program_{self.date_timestamp}_{self.job_timestamp}.qasm"
        )
        cqasm_v3_program_path = (
            Path(self.job_dir)
            / f"cqasm_v3_program_{self.date_timestamp}_{self.job_timestamp}.cq"
        )
        with open(qasm3_program_path, 'w') as f:
            f.write(qasm3_program)
        with open(cqasm_v3_program_path, 'w') as f:
            f.write(cqasm_v3_program)

        fig1 = self.qc.draw('mpl', scale=1.3)
        fig1.suptitle(f'\n{self.date_timestamp}_{self.job_timestamp}\nTranspiled quantum circuit\nCircuit name: {self.circuit_name}\nJob ID: {self.job_id}\n',
                      x = 0.5, y = 0.99, fontsize=16)
        fig1.supxlabel(f'Circuit depth: {self.circuit_depth}', x = 0.5, y = 0.06, fontsize=18)
        circuit_fig_path = (
            Path(self.job_dir)
            / f"quantum_circuit_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig1.savefig(circuit_fig_path)

    def store_raw_data(self,
                       job: QIJob,
                       job_idx: int):
        """
        This instance method stores the job raw data results in an HDF5
        file format.

        It is important here to mention the conventions: assume that a quantum
        circuit contains M number of mid-circuit measurements, and that we execute
        it with N number of shots. The raw data (raw_data) is returned in the form
        of a list of bitstrings, where each entry of that list represents a circuit shot,
        and each bitstring contains all measurement outcomes from each mid-circuit
        measurement block.

        In the Tuna backends, the rightmost bit in the bitstring is the first measurement
        outcome, while the leftmost bit is the final measurement outcome,
        e.g. for a bit register of size 3, the bitstring '001' means that the first
        measurement outcome was '1' and was stored in the bit c0, while the second and
        third measurement outcomes were '0' and were stored in the bits c1 and c2.

        In the HDF5 file, we reverse the order of the bitstring for clarity. The file
        contains a 2D array of N rows, representing each shot, and M columns, representing
        all mid-circuit measurement outcomes. Column 0 in this case represents the first
        measurement outcome, while column M-1 represents the final measurement outcome
        for a particular measurement shot. 

        Args:
            job (QIJob):
                The user already-submitted job (project) object.

            job_idx (int):
                The job index for all jobs contained within the project.
                While a project may contain a certain number of jobs, e.g. N,
                it is generally true that the execution of all these jobs
                in the Quantum Inspire platform is not sequential with respect
                to the order with which those jobs were created.
                Therefore, job_idx is being utilized for clarity when storing
                the data, so that it follows the sequence with which the jobs were
                created.
        """

        raw_data = job.circuits_run_data[job_idx].results.raw_data
        job_raw_data = []

        for circuit_shot_idx in range(len(raw_data)):

            raw_data_row = [int(raw_data[circuit_shot_idx][digit_idx]) for digit_idx in range(len(raw_data[0]))]
            raw_data_row_reversed = raw_data_row[::-1] # because results are printed reversed
            job_raw_data.append(raw_data_row_reversed)

        job_raw_data = np.array(job_raw_data, dtype=np.int8)

        hdf5_file_dir = (
            Path(self.job_dir)
            / f"raw_data_{self.date_timestamp}_{self.job_timestamp}.hdf5"
        )
        with h5py.File(hdf5_file_dir, 'w') as file:
            file.create_dataset('Experimental Data/Data', data=job_raw_data, compression="gzip")

class RetrieveProjectRecord:
    """
    This class is responsible for retrieving a single job record
    within the user's local 'Documents' directory, by providing as inputs
    the appropriate job timestamp, name, and Job ID. Those three variables
    uniquely identify a single job in the projects directory.

    It is meant to mimic 'result = job.result()', so that when the user runs

    loaded_result = RetrieveProjectRecord(timestamp, project_name, job_id)

    then they can obtain the measurement counts and raw data as

    loaded_result.get_counts()
    loaded_result.get_memory()
    """

    def __init__(self,
                 timestamp: str,
                 project_name: str,
                 job_id: str):
        """
        Args:
            timestamp (str):
                The job complete timestamp, to be parsed as 'YYYYMMDD_HHMMSS'.
            
            project_name (str):
                The original project name.
            
            job_id (str):
                The Job ID, as this appears also in the Quantum Inspire platform.
        """
        
        self.timestamp = timestamp
        self.project_name = project_name
        self.job_id = job_id

        date_timestamp = self.timestamp.split('_')[0]
        job_timestamp = self.timestamp.split('_')[1]

        project_dir = (
            Path.home() / "Documents" / "QuantumInspireProjects" / date_timestamp
            / f"{job_timestamp}_{self.project_name}"
        )

        self.job_dir = None
        jobs_folders = [p for p in project_dir.iterdir() if p.is_dir()]
        for dir_entry in jobs_folders:
            if self.job_id in dir_entry.name:
                self.job_dir = dir_entry
        if self.job_dir == None:
            raise ValueError(f'No files found for timestamp: {timestamp}, project name: {project_name}, and Job ID: {job_id}')

        self.retrieve_qc()
        self.get_counts()
        self.get_memory()

    def retrieve_qc(self):
        """
        This instance method retrieves the QuantumCircuit object of the job.
        """

        qasm3_file_path = (
            Path(self.job_dir)
            / f"qasm3_program_{self.timestamp}.qasm"
        )
        self.qc = qasm3.load(qasm3_file_path)

    def get_counts(self):
        """
        This instance method retrieves the job counts in a dictionary format.
        """

        json_file_path = (
            Path(self.job_dir)
            / f"job_result_{self.timestamp}.json"
        )

        with open(json_file_path, 'r') as file:
            json_data = json.load(file)

        counts = json_data['Counts']
        return counts

    def get_memory(self,
                   dummy_circuit_nr: int = None):
        """
        This instance method retrieves the job raw data in a list of string
        bitstrings format.

        For jobs were the variable 'memory' was set to False,
        e.g. job = backend.run(qc, shots=nr_shots, memory = False)
        this instance method will return an empty list.

        Args:
            dummy_circuit_nr (int):
                This is a dummy variable which does not affect the instance method.
                It exists so that the instance method mimics result = job.result().get_memory(),
                which does have the 'circuit_nr' as an input.
                In result = job.result().get_memory(circuit_nr), since the job object in
                reality is a project and can in principle contain multiple jobs, the variable
                circuit_nr is used to identify uniquely a single circuit tied to a single job.
                
                Since when instantiating the RetrieveProjectRecord class we have already
                identified a single job, the circuit_nr would have no meaning.
                Still, we use here a dummy circuit_nr variable for compatibility purposes with
                respect to other functions used in other modules.
        """

        hdf5_file_dir = (
            Path(self.job_dir)
            / f"raw_data_{self.timestamp}.hdf5"
        )

        try:
            with h5py.File(hdf5_file_dir, "r") as f:
                hdf5_data = f["Experimental Data"]["Data"][()]
            raw_shots = []
            for shot_idx in range(len(hdf5_data)):
                shot_string = ''.join(map(str, hdf5_data[shot_idx][::-1]))
                raw_shots.append(shot_string)
            return raw_shots

        except:
            return []