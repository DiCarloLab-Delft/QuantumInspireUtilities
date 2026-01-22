import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from qiskit import transpile
from qiskit import qasm3
from qiskit_algorithms.optimizers import SPSA
from qi_utilities.utility_functions.data_handling import StoreProjectRecord
from qi_utilities.utility_functions.raw_data_processing import (get_multi_counts, get_multi_probs,
                                                                observable_expectation_values_Z_basis)
from qi_utilities.utility_functions.readout_correction import get_ro_corrected_multi_probs, measure_ro_assignment_matrix

class ExecuteVQE:
    
    def __init__(self,
                 variational_qc,
                 hamiltonian,
                 hamiltonian_units,
                 backend,
                 qubit_list,
                 basis_gates,
                 nr_shots = 2**11,
                 maxiter = 200,
                 project_name = None):
        
        self.termination_status = False
        self.variational_qc = variational_qc
        self.hamiltonian = hamiltonian
        self.hamiltonian_units = hamiltonian_units
        self.backend = backend
        self.nr_qubits = len(qubit_list)
        self.qubit_list = qubit_list
        self.basis_gates = basis_gates
        self.nr_shots = nr_shots
        self.maxiter = maxiter
        self.project_name = project_name

        if self.project_name == None:
            self.project_name = f'VQE_{self.nr_qubits}_Qubits'

        self.job_directories = []
        self.var_parameter_list = []
        self.output_energies_list = []

        self.create_project_directory()
        self._init_plot()  # Initialize live plot
        self.store_variational_circuit()
        self.run()

    def _init_plot(self):
        """Create a single live figure for VQE energy."""
        import matplotlib
        self.init_mpl_backend = matplotlib.get_backend()
        matplotlib.use("Qt5Agg", force=True)
        plt.ion()

        lower_bound = 0
        coeffs = np.real(self.hamiltonian.coeffs)
        for coeff in coeffs:
            lower_bound -= np.abs(coeff)

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Iteration, k")
        self.ax.set_ylabel(f"Energy [{self.hamiltonian_units}]")
        self.ax.set_title(f"{self.date_timestamp}_{self.project_timestamp}\nVQE Energy Convergence")
        self.ax.set_xlim(-0.1, 2*self.maxiter+1.1)
        self.ax.set_ylim(1.1*lower_bound, 0.5*np.abs(lower_bound))
        self.line, = self.ax.plot([], [], 'b-o', alpha=0.7,
                                  color='C0', linewidth=2, label='Data points')
        self.ax.axhline(lower_bound, color='black',
                        linestyle='--', alpha=0.7, linewidth=2, label='Lower bound')
        self.ax.legend()

        self.fig.canvas.manager.set_window_title(f"{self.date_timestamp}_{self.project_timestamp} VQE Energy Live Plot")
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_plot(self):
        """Update the live plot with current energies."""
        self.line.set_xdata(range(1, len(self.output_energies_list)+1))
        self.line.set_ydata(self.output_energies_list)
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # in order to process live the newest data points

    def create_project_directory(self):

        timestamp = datetime.now()
        self.date_timestamp = timestamp.strftime("%Y%m%d")
        self.project_timestamp = timestamp.strftime("%H%M%S")

        self.project_dir = (
            Path.home() / "Documents" / "QuantumInspireProjects" / self.date_timestamp
            / f"{self.project_timestamp}_{self.project_name}"
        )
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def store_project_json(self):
        """
        This instance method stores job (project) related metadata within
        the project directory in a JSON format.
        """
        
        general_dict = {}
        general_dict['Project name'] = self.project_name
        general_dict['Project timestamp'] = f"{self.date_timestamp}_{self.project_timestamp}"
        general_dict['Termination status'] = self.termination_status

        general_dict['Backend info'] = {}
        general_dict['Experiment metadata'] = {}
        general_dict['Experiment data'] = {}

        general_dict['Backend info']['Backend name'] = self.backend.name
        general_dict['Backend info']['Backend number of qubits'] = self.backend.num_qubits
        general_dict['Backend info']['Backend maximum allowed shots'] = self.backend.max_shots

        general_dict['Experiment metadata']['Hamiltonian operator'] = {}
        general_dict['Experiment metadata']['Hamiltonian operator']['Paulis'] = self.hamiltonian.paulis.to_labels()
        general_dict['Experiment metadata']['Hamiltonian operator']['Coefficients (real)'] = np.real(self.hamiltonian.coeffs).tolist()
        general_dict['Experiment metadata']['Qubits used'] = self.qubit_list
        general_dict['Experiment metadata']['Number of shots'] = self.nr_shots
        general_dict['Experiment metadata']['Readout assignment matrix'] = self.ro_assignment_matrix.tolist()
        general_dict['Experiment metadata']['Optimizer maxiter'] = self.maxiter
        general_dict['Experiment metadata']['Jobs directories'] = self.job_directories

        general_dict['Experiment data']['var_parameters'] = self.var_parameter_list
        general_dict['Experiment data']['output_energies'] = self.output_energies_list

        file_path = (
            Path(self.project_dir)
            / f"project_data_{self.date_timestamp}_{self.project_timestamp}.json"
        )
        with open(file_path, 'w') as file:
            json.dump(general_dict, file, indent=3)

    def store_variational_circuit(self):

        self.circuit_name = self.variational_qc.name
        self.num_qubits = self.variational_qc.to_instruction().num_qubits
        self.num_clbits = self.variational_qc.to_instruction().num_clbits
        self.circuit_depth = self.variational_qc.depth()

        fig1 = self.variational_qc.draw('mpl', scale=1.3)
        fig1.suptitle(f'\n{self.date_timestamp}_{self.project_timestamp}\nVariational quantum circuit\nCircuit name: {self.circuit_name}\nNr variational parameters: {self.variational_qc.num_parameters}',
                      x = 0.5, y = 0.99, fontsize=16)
        fig1.supxlabel(f'Circuit depth: {self.circuit_depth}', x = 0.5, y = 0.06, fontsize=18)
        circuit_fig_path = (
            Path(self.project_dir)
            / f"variational_circuit_{self.date_timestamp}_{self.project_timestamp}.png"
        )
        fig1.savefig(circuit_fig_path)
        plt.close(fig1)

    def make_ansatz(self,
                    var_parameters):
        
        ansatz_circuit = self.variational_qc
        ansatz_circuit = self.variational_qc.assign_parameters(var_parameters)
        return ansatz_circuit
    
    def translate_to_Z_basis(self,
                             pauli_term: str):

        pauli_term = pauli_term.replace("X", "Z")
        pauli_term = pauli_term.replace("Y", "Z")
        return pauli_term
    
    def calculate_energy(self,
                         result):
    
        raw_data_shots = result.get_memory()
        raw_data_counts = get_multi_counts(raw_data_shots, len(self.qubit_list))
        raw_data_probs = get_multi_probs(raw_data_counts)
        ro_corrected_probs = get_ro_corrected_multi_probs(raw_data_probs, self.ro_assignment_matrix, self.qubit_list)

        output_energy = 0
        for term_idx in range(len(self.hamiltonian)):

            pauli_term = self.hamiltonian[term_idx].paulis[0].to_label()
            pauli_term_in_Z = self.translate_to_Z_basis(pauli_term)
            observable_value = observable_expectation_values_Z_basis([ro_corrected_probs[term_idx]], pauli_term_in_Z)[0]
            coeff = np.real(self.hamiltonian[term_idx].coeffs[0])
            output_energy += coeff * observable_value

        return output_energy
    
    def cost_function(self,
                      var_parameters):
    
        qc_instance = self.make_ansatz(var_parameters)
        qc_instance_transpiled = transpile(qc_instance,
                                self.backend,
                                initial_layout=self.qubit_list,
                                basis_gates=self.basis_gates)
        
        job = self.backend.run(qc_instance_transpiled, shots=self.nr_shots, memory=True)
        result = job.result(timeout = 6 * 600)
        job_record = StoreProjectRecord(job, silent=True)
        output_energy = self.calculate_energy(result)

        self.job_directories.append(f"{job_record.job_dir.parts[-3]}_{job_record.job_dir.parts[-2]}_{job_record.job_dir.parts[-1]}")
        self.var_parameter_list.append(var_parameters.tolist())
        self.output_energies_list.append(output_energy)

        self.store_project_json()
        self._update_plot()  # <-- live plot update

        return output_energy

    def run(self):

        initial_point = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=self.variational_qc.num_parameters
        )
        self.ro_assignment_matrix = measure_ro_assignment_matrix(self.backend,
                                                                 self.qubit_list,
                                                                 self.nr_shots)

        SPSA_optimizer = SPSA(
            maxiter=self.maxiter,
            perturbation=0.1 * np.pi,
            learning_rate=0.05 * np.pi,
            last_avg = 20,
            blocking=True
            )

        self.result = SPSA_optimizer.minimize(self.cost_function,
                                        x0 = initial_point)
        
        self.termination_status = True
        self.store_project_json()

        vqe_fig_path = (
            Path(self.project_dir)
            / f"vqe_run_{self.date_timestamp}_{self.project_timestamp}.png"
        )
        self.fig.savefig(vqe_fig_path, dpi=300)

        import matplotlib
        matplotlib.use(self.init_mpl_backend, force=True)