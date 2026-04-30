import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from qiskit import QuantumCircuit
from qi_utilities.measurements.fitting_functions import exp_decay_func, cos_func, damped_osc_func
from qi_utilities.measurements.base_classes import BaseMeasurement


class RabiMeasurement(BaseMeasurement):

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 rotation_angles: np.array = np.linspace(0, 2*np.pi, num=29),
                 directory: str = None):
        """
        Args:
            rotation_angles (np.array):
                The different rotation angles for the Rabi oscillation
                expressed in radians [rad].
            
        """
        
        if len(rotation_angles) < 4:
            raise ValueError("rotation_angles must have at least 4 entries!")
        
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   rotation_angles)
        
        super().__init__(backend,
                         qubit_list,
                         qc,
                         num_shots,
                         directory)
        
        self._data_analysis(rotation_angles)
        
    def _quantum_circuit(self,
                        backend,
                        qubit_list: list,
                        rotation_angles: np.array = np.linspace(0, 2*np.pi, num=29)):

        bit_idx = 0
        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            len(rotation_angles)*len(qubit_list),
                            name=f'Rabi_{self.qubit_labels}')

        for step_idx in range(len(rotation_angles)):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.rx(rotation_angles[step_idx], qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()
            
        return qc
    
    def _data_analysis(self,
                       rotation_angles: np.array):
        
        fig, ax = plt.subplots(figsize=(18, 5), dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
                                     for entry in range(len(rotation_angles))]
            params, covariance = curve_fit(cos_func,
                                           rotation_angles,
                                           probabilities_excited)
            a_fit, b_fit, c_fit, d_fit = params
            cosine_fit = cos_func(rotation_angles, a_fit, b_fit, c_fit, d_fit)

            ax.scatter(rotation_angles,
                       probabilities_excited,
                       label=f'{self.qubit_labels[qubit_idx]}',
                       alpha=0.6,
                       color = f'C{qubit_idx}')
            ax.plot(rotation_angles,
                    cosine_fit,
                    alpha=0.6,
                    color = f'C{qubit_idx}')
        ax.set_xlabel('Applied rotation')
        ax.set_ylabel(r'$P(|1\rangle)$')
        ax.set_title(f'Rabi oscillation\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
        
        labels = []
        for step_idx in range(len(rotation_angles)):
            angle_in_degrees = (360 / (2 * np.pi)) * rotation_angles[step_idx]
            step_label = f"rx{round(angle_in_degrees)}"
            labels.append(step_label)
        label_locs = rotation_angles
        ax.set_xticks(label_locs)
        ax.set_xticklabels(labels, rotation=65)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid()
        rabi_fig_path = (
            Path(self.record.project_dir)
            / f"rabi_plots_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig.savefig(rabi_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Rotation angles [rad]"] = rotation_angles
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        json_file_path = (
            Path(self.record.project_dir)
            / f"rabi_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)


class T1_Measurement(BaseMeasurement):

    def __init__(self,
                 backend,
                 qubit_list: list,
                 measurement_times: np.array,
                 num_shots: int,
                 directory: str = None):
        """
        Args:
            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        
        """
        
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   measurement_times)
        
        super().__init__(backend,
                         qubit_list,
                         qc,
                         num_shots,
                         directory)
        
        self._data_analysis(measurement_times)
        

    def _quantum_circuit(self,
                        backend,
                        qubit_list: list,
                        measurement_times: np.array):

        cycle_time = 20e-9 # cycle time limitation imposed by the Tuna backend internal software
        total_time = measurement_times[-1]
        num_points = len(measurement_times)
        dt = total_time / num_points
        bit_idx = 0

        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            num_points*len(qubit_list),
                            name=f'T1_{self.qubit_labels}')
        
        for time_idx in range(num_points):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.x(qubit_idx) # qubit initialization to the |1> state
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.delay(duration = time_idx * int((dt / cycle_time)),
                        qarg = qubit_idx) # delay in units of cycle_time
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()

        return qc
    
    def _data_analysis(self,
                       measurement_times: np.array):
        
        self.T1_values = {}
        
        fig, ax = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            params, covariance = curve_fit(exp_decay_func,
                                           measurement_times,
                                           probabilities_excited,
                                           bounds=(
                                                [0, 0],   # lower bounds
                                                [np.inf, 2.0]  # upper bounds
                                            )
                                           )
            tau_fit, amplitude_fit = params
            exponential_fit = exp_decay_func(measurement_times, tau_fit, amplitude_fit)
            self.T1_values[f'{self.qubit_labels[qubit_idx]} [us]'] = 1e6 * tau_fit

            ax.scatter(1e6*measurement_times,
                       probabilities_excited,
                       label=f'{self.qubit_labels[qubit_idx]}: T1 = {1e6 * tau_fit:.1f} μs',
                       alpha=0.6,
                       color = f'C{qubit_idx}')
            ax.plot(1e6*measurement_times,
                    exponential_fit,
                    alpha=0.6,
                    color = f'C{qubit_idx}')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel(r'$P(|1\rangle)$')
        ax.set_title(f'T1 measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        T1_fig_path = (
            Path(self.record.project_dir)
            / f"T1_plots_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig.savefig(T1_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T1 values"] = self.T1_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T1_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)


class T2_RamseyMeasurement(BaseMeasurement):

    def __init__(self,
                 backend,
                 qubit_list: list,
                 measurement_times: np.array,
                 num_shots: int,
                 artificial_detuning: float = None,
                 directory: str = None):
        """
        Args:
            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        
        """
        
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   measurement_times,
                                   artificial_detuning)
        
        super().__init__(backend,
                         qubit_list,
                         qc,
                         num_shots,
                         directory)
        
        self._data_analysis(measurement_times)
        

    def _quantum_circuit(self,
                         backend,
                         qubit_list: list,
                         measurement_times: np.array,
                         artificial_detuning: float = None):

        if artificial_detuning is None:
            self.artificial_detuning = 5 / measurement_times[-1]
        else:
            self.artificial_detuning = artificial_detuning

        cycle_time = 20e-9 # cycle time limitation imposed by the Tuna backend internal software
        total_time = measurement_times[-1]
        num_points = len(measurement_times)
        dt = total_time / num_points
        bit_idx = 0

        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            num_points*len(qubit_list),
                            name=f'T2_Ramsey_{self.qubit_labels}')

        for time_idx in range(num_points):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            # qubit initialization to the |+> state
            for qubit_idx in qubit_list:
                qc.rx(np.pi/2, qubit_idx)
            qc.barrier()

            # delay in units of cycle_time
            for qubit_idx in qubit_list:
                qc.delay(duration = time_idx * int((dt / cycle_time)),
                        qarg = qubit_idx)
            qc.barrier()

            # adding the artificial detuning which rotates the qubit state
            for qubit_idx in qubit_list:
                qc.rz(- (2*np.pi) * self.artificial_detuning * (time_idx*dt),
                    qubit_idx)
            qc.barrier()

            # rotating qubit state (ideally to the |1> state)
            for qubit_idx in qubit_list:
                qc.rx(np.pi/2, qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()

        return qc
    
    def _data_analysis(self,
                       measurement_times: np.array):
        
        self.T2_ramsey_values = {}
        
        fig, ax = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            p0 = [
                    measurement_times[-1] / 2,
                    self.artificial_detuning,
                    0.0,
                    max(probabilities_excited),
                    0.0,
                    0.0
                ]
            params, covariance = curve_fit(damped_osc_func,
                                           measurement_times,
                                           probabilities_excited,
                                           p0 = p0,
                                           bounds = (
                                                [0, self.artificial_detuning/2, -np.pi,   0, 0, 0],
                                                [5*measurement_times[-1], 2*self.artificial_detuning, np.pi, 4.0, np.inf, np.inf]
                                            )
                                           )
            tau_fit, frequency_fit, phase_fit, amplitude_fit, osc_offset_fit, exp_offset_fit = params
            damped_oscillation_fit = damped_osc_func(measurement_times, tau_fit, frequency_fit,
                                                     phase_fit, amplitude_fit,
                                                     osc_offset_fit, exp_offset_fit)
            self.T2_ramsey_values[f'{self.qubit_labels[qubit_idx]} [us]'] = 1e6 * tau_fit

            ax.scatter(1e6*measurement_times,
                       probabilities_excited,
                       label=f'{self.qubit_labels[qubit_idx]}: T2* = {1e6 * tau_fit:.1f} μs',
                       alpha=0.6,
                       color = f'C{qubit_idx}')
            ax.plot(1e6*measurement_times,
                    damped_oscillation_fit,
                    alpha=0.6,
                    color = f'C{qubit_idx}')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel(r'$P(|1\rangle)$')
        ax.set_title(f'T2 Ramsey measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        T1_fig_path = (
            Path(self.record.project_dir)
            / f"T2_Ramsey_plots_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig.savefig(T1_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Artificial detuning [Hz]"] = self.artificial_detuning
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T2 Ramsey values"] = self.T2_ramsey_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T2_Ramsey_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)



def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj