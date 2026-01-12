import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageFilter
from zoneinfo import ZoneInfo
from qiskit import qasm3
from qiskit_quantuminspire.cqasm import dumps

class StoreJobRecord:

    def __init__(self,
                 job):
        
        self.create_project_directory(job)
        self.obtain_backend_data(job)
        self.create_project_json()
        for job_idx in range(len(job.circuits_run_data)):
            self.create_job_directory(job, job_idx)
            self.store_job_result(job, job_idx)
            self.store_circuit_data(job, job_idx)
            if self.raw_data_memory == True:
                self.store_raw_data(job, job_idx)

        return print("Success!")

    def create_project_directory(self,
                                 job):
        
        self.timestamp_utc = job.circuits_run_data[0].results.created_on # actually when the job finished
        self.timestamp = self.timestamp_utc.astimezone()
        self.date_timestamp = self.timestamp.strftime("%Y%m%d")
        self.job_0_timestamp = self.timestamp.strftime("%H%M%S")

        self.project_name = job.program_name
        self.project_dir = (
            Path.home() / "Documents" / "QuantumInspireProjects" / self.date_timestamp
            / f"{self.job_0_timestamp}_{self.project_name}"
        )
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def obtain_backend_data(self,
                            job):

        self.backend_name = job.backend().name
        self.backend_nr_qubits = job.backend().num_qubits
        self.backend_operations = []
        for entry in range(len(job.backend().operations)):
            self.backend_operations.append(str(job.backend().operations[entry]))
        self.backend_max_shots = job.backend().max_shots

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

    def create_project_json(self):
        
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
                             job,
                             job_idx):
        
        self.timestamp_utc = job.circuits_run_data[job_idx].results.created_on # actually when the job finished
        self.timestamp = self.timestamp_utc.astimezone()
        self.date_timestamp = self.timestamp.strftime("%Y%m%d")
        self.job_timestamp = self.timestamp.strftime("%H%M%S")
        self.job_id = job.circuits_run_data[job_idx].results.job_id

        self.job_dir = (
            Path.home()
            / "Documents" / "QuantumInspireProjects" / self.date_timestamp
            / f"{self.job_0_timestamp}_{self.project_name}"
            / f"job_idx_{job_idx}__job_id_{self.job_id}"
        )
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def store_job_result(self,
                         job,
                         job_idx):
        
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

    def store_circuit_data(self,
                           job,
                           job_idx):

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
        fig1.suptitle(f'\n{self.date_timestamp}_{self.job_timestamp}\nTranspiled quantum circuit\nJob ID: {self.job_id}\n',
                      x = 0.5, y = 0.99, fontsize=16)
        fig1.supxlabel(f'Circuit depth: {self.circuit_depth}', x = 0.5, y = 0.06, fontsize=18)
        circuit_fig_path = (
            Path(self.job_dir)
            / f"quantum_circuit_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig1.savefig(circuit_fig_path)


    def store_raw_data(self,
                       job,
                       job_idx):

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