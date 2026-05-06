import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit, minimize
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
from qi_utilities.utility_functions.circuit_modifiers import apply_rotations_from_list
from qi_utilities.measurements.fitting_functions import exp_decay_func, cos_func, damped_osc_func
from qi_utilities.utility_functions.raw_data_processing import translate_to_Z_basis, observable_expectation_values_Z_basis
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
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         directory=directory)
        
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
        
        self.rabi_amplitudes = {}
        
        fig_all, ax_all = plt.subplots(figsize=(18, 5), dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(rotation_angles))]
            params, covariance = curve_fit(cos_func,
                                           rotation_angles,
                                           probabilities_excited)
            a_fit, b_fit, c_fit, d_fit = params
            cosine_fit = cos_func(rotation_angles, a_fit, b_fit, c_fit, d_fit)
            self.rabi_amplitudes[f'{self.qubit_labels[qubit_idx]} [a.u.]'] = cos_func(np.pi, a_fit, b_fit, c_fit, d_fit)

            fig, ax = plt.subplots(figsize=(18, 5), dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.scatter(rotation_angles,
                        probabilities_excited,
                        label=f'Qubit {self.qubit_labels[qubit_idx]}',
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.plot(rotation_angles,
                        cosine_fit,
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.set_xlabel('Applied rotation')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(f'Rabi oscillation\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
        
                labels = []
                for step_idx in range(len(rotation_angles)):
                    angle_in_degrees = (360 / (2 * np.pi)) * rotation_angles[step_idx]
                    step_label = f"rx{round(angle_in_degrees)}"
                    labels.append(step_label)
                label_locs = rotation_angles
                ax_obj.set_xticks(label_locs)
                ax_obj.set_xticklabels(labels, rotation=65)
                ax_obj.set_ylim(-0.05, 1.05)
                ax_obj.legend()
                ax_obj.grid(True)
            rabi_fig_path = (
                Path(self.record.project_dir)
                / f"rabi_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(rabi_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        rabi_all_fig_path = (
            Path(self.record.project_dir)
            / f"rabi_plot_ALL_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig_all.savefig(rabi_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Rotation angles [rad]"] = rotation_angles
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["Rabi amplitudes"] = self.rabi_amplitudes
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
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         directory=directory)
        
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
        
        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_n_qubits[qubit_idx][entry]['1'] \
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

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.scatter(1e6*measurement_times,
                        probabilities_excited,
                        label=f'{self.qubit_labels[qubit_idx]}: T1 = {1e6 * tau_fit:.1f} μs',
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.plot(1e6*measurement_times,
                        exponential_fit,
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.set_xlabel('Time (μs)')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(f'T1 measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
                ax_obj.set_ylim(-0.05, 1.05)
                ax_obj.legend()
            T1_fig_path = (
                Path(self.record.project_dir)
                / f"T1_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(T1_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        T1_all_fig_path = (
            Path(self.record.project_dir)
            / f"T1_plot_ALL_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig_all.savefig(T1_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_n_qubits[qubit_idx] \
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
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         directory=directory)
        
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

            # qubit initialization to the |-i> state
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
        
        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            p0 = [
                    measurement_times[-1] / 2,
                    self.artificial_detuning,
                    0.0,
                    0.5*(max(probabilities_excited) - min(probabilities_excited)),
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

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.scatter(1e6*measurement_times,
                        probabilities_excited,
                        label=f'{self.qubit_labels[qubit_idx]}: T2* = {1e6 * tau_fit:.1f} μs',
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.plot(1e6*measurement_times,
                        damped_oscillation_fit,
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.set_xlabel('Time (μs)')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(f'T2 Ramsey measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
                ax_obj.set_ylim(-0.05, 1.05)
                ax_obj.legend()
            T2_ramsey_fig_path = (
                Path(self.record.project_dir)
                / f"T2_Ramsey_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(T2_ramsey_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        T2_ramsey_all_fig_path = (
            Path(self.record.project_dir)
            / f"T2_Ramsey_plot_ALL_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig_all.savefig(T2_ramsey_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Artificial detuning [Hz]"] = self.artificial_detuning
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T2 Ramsey values"] = self.T2_ramsey_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T2_Ramsey_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)


class T2_EchoMeasurement(BaseMeasurement):

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
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         directory=directory)
        
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
                            name=f'T2_Echo_{self.qubit_labels}')

        for time_idx in range(num_points):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            # qubit initialization to the |-i> state
            for qubit_idx in qubit_list:
                qc.rx(np.pi/2, qubit_idx)
            qc.barrier()
    
            # delay/2 in units of cycle_time
            for qubit_idx in qubit_list:
                qc.delay(duration = time_idx * int(((dt/2) / cycle_time)),
                        qarg = qubit_idx)
            qc.barrier()
            # applying the echo pulse
            for qubit_idx in qubit_list:
                qc.rx(np.pi, qubit_idx)
            qc.barrier()
            # delay/2 in units of cycle_time
            for qubit_idx in qubit_list:
                qc.delay(duration = time_idx * int(((dt/2) / cycle_time)),
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
        
        self.T2_echo_values = {}
        
        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            p0 = [
                    measurement_times[-1] / 2,
                    self.artificial_detuning,
                    np.pi,
                    0.5*(max(probabilities_excited) - min(probabilities_excited)),
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
            self.T2_echo_values[f'{self.qubit_labels[qubit_idx]} [us]'] = 1e6 * tau_fit

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.scatter(1e6*measurement_times,
                        probabilities_excited,
                        label=f'{self.qubit_labels[qubit_idx]}: T2 echo = {1e6 * tau_fit:.1f} μs',
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.plot(1e6*measurement_times,
                        damped_oscillation_fit,
                        alpha=0.6,
                        color = f'C{qubit_idx}')
                ax_obj.set_xlabel('Time (μs)')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(f'T2 echo measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
                ax_obj.set_ylim(-0.05, 1.05)
                ax_obj.legend()
            T2_echo_fig_path = (
                Path(self.record.project_dir)
                / f"T2_echo_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(T2_echo_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        T2_echo_all_fig_path = (
            Path(self.record.project_dir)
            / f"T2_echo_plot_ALL_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig_all.savefig(T2_echo_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Artificial detuning [Hz]"] = self.artificial_detuning
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T2 echo values"] = self.T2_echo_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T2_echo_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)


class AllXYMeasurement(BaseMeasurement):

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 directory: str = None):
        
        qc = self._quantum_circuit(backend,
                                   qubit_list)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         directory=directory)
        
        self._data_analysis()
        
    def _quantum_circuit(self,
                         backend,
                         qubit_list):

        AllXY_syndromes = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
                            ['rx180', 'ry180'], ['ry180', 'rx180'],
                            ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
                            ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
                            ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
                            ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
                            ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
                            ['ry90', 'ry90']]


        repetitions=2
        bit_idx = 0
        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            repetitions*len(AllXY_syndromes)*len(qubit_list),
                            name=f'AllXY_{self.qubit_labels}')

        for syndrome_idx, syndrome in enumerate(AllXY_syndromes):
            for repetition_idx in range(repetitions):
                for qubit_idx in qubit_list:
                    qc.reset(qubit_idx)
                qc.barrier()
                for qubit_idx in qubit_list:
                    for pulse_idx in range(len(syndrome)):
                        if '90' in syndrome[pulse_idx]:
                            rotation_angle = np.pi/2
                        if '180' in syndrome[pulse_idx]:
                            rotation_angle = np.pi
                        if 'i' in syndrome[pulse_idx]:
                            qc.id(qubit_idx)
                            continue
                        if 'rx' in syndrome[pulse_idx]:
                            qc.rx(rotation_angle, qubit_idx)
                            continue
                        if 'ry' in syndrome[pulse_idx]:
                            qc.ry(rotation_angle, qubit_idx)
                            continue
                qc.barrier()

                for qubit_idx in qubit_list:
                    qc.measure(qubit = qubit_idx, cbit = bit_idx)
                    bit_idx += 1
                qc.barrier()

        return qc
    
    def _data_analysis(self):
        
        AllXY_syndrome_labels = ['II', 'XX', 'YY', 'XY', 'YX', 'xI', 'yI', 'xy',
                                 'yx', 'xY', 'yX', 'Xy', 'Yx', 'xX', 'Xx', 'yY',
                                 'Yy', 'XI', 'YI', 'xx', 'yy']
        self.allxy_deviations = {}

        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(2*len(AllXY_syndrome_labels))]
            
            ideal_data = np.concatenate((0 * np.ones(10), 0.5 * np.ones(24),
                            np.ones(8)))
            data_error = probabilities_excited - ideal_data
            deviation_total = np.mean(abs(data_error))
            self.allxy_deviations[f'{self.qubit_labels[qubit_idx]} deviation'] = deviation_total

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.plot(np.arange(start=0,
                                      stop=2*len(AllXY_syndrome_labels),
                                      step=1),
                        probabilities_excited,
                        label=f'Qubit {self.qubit_labels[qubit_idx]}',
                        alpha=0.6,
                        color = f'C{qubit_idx}',
                        linestyle='-', marker='o')
                ax_obj.set_xlabel('Syndrome')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(f'AllXY measurement\n{self.backend.name} processor\nQubit list: {self.qubit_labels}\n{self.record.date_timestamp}_{self.record.job_timestamp}')
        
                labels = []
                for label_idx in range(len(AllXY_syndrome_labels)):
                    labels.append(AllXY_syndrome_labels[label_idx])
                label_locs = np.arange(start=1,
                                       stop=2*len(AllXY_syndrome_labels),
                                       step=2)
                ax_obj.set_xticks(label_locs)
                ax_obj.set_xticklabels(labels, rotation=65)
                ax_obj.set_ylim(-0.05, 1.05)
            ax.plot(np.arange(start=0,
                              stop=2*len(AllXY_syndrome_labels),
                              step=1),
                    ideal_data,
                    label=f'Ideal | Deviation: {deviation_total:.5f}',
                    alpha=0.6,
                    color = 'red')
            ax.legend()
            allxy_fig_path = (
                Path(self.record.project_dir)
                / f"allxy_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(allxy_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        ax_all.legend()
        allxy_all_fig_path = (
            Path(self.record.project_dir)
            / f"allxy_plot_ALL_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
        )
        fig_all.savefig(allxy_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["AllXY deviations"] = self.allxy_deviations
        json_file_path = (
            Path(self.record.project_dir)
            / f"allxy_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    
class BellStateMeasurement(BaseMeasurement):

    def __init__(self,
                backend,
                qubit_pairs: list[list[int]],
                bell_state: str,
                num_shots: int,
                directory: str = None):
        
        qubit_list = []
        for qubit_pair in qubit_pairs:
            for qubit_idx in qubit_pair:
                if qubit_idx in qubit_list:
                    raise ValueError("A single qubit cannot be used in multiple pairs at the same time!")
                else:
                    qubit_list.append(qubit_idx)
        self.tomography_bases = [
                    'XX', 'YX', 'ZX', 'XY', 'YY',
                    'ZY', 'XZ', 'YZ', 'ZZ'
                ]
        self.single_qubit_terms = ['IX', 'IY', 'IZ', 'XI', 'YI', 'ZI']
        
        qc = self._quantum_circuit(backend,
                                qubit_pairs,
                                bell_state)

        super().__init__(backend=backend,
                        qubit_list=qubit_list,
                        qc=qc,
                        num_shots=num_shots,
                        n_qubit_routine=2,
                        qubit_groups = qubit_pairs,
                        directory=directory)
        
        self._data_analysis(qubit_pairs,
                            bell_state)
        

    def _quantum_circuit(self,
                        backend,
                        qubit_pairs: list[list[int]],
                        bell_state: str):

        def create_bell_states(qc: QuantumCircuit,
                            qubit_pairs: list[list[int]],
                            bell_state: str):
            
            bell_state_rotations = {
                "phi_plus": [0.0, 0.0],
                "phi_minus": [np.pi, 0.0],
                "psi_plus": [0.0, np.pi],
                "psi_minus": [np.pi, np.pi]
            }

            for qubit_pair in qubit_pairs:
                for qubit_idx in qubit_pair:        
                    qc.reset(qubit_idx)
            for qubit_pair in qubit_pairs:
                qc.rx(bell_state_rotations[bell_state][0], qubit_pair[0])
                qc.rx(bell_state_rotations[bell_state][1], qubit_pair[1])
            qc.barrier()
            for qubit_pair in qubit_pairs:
                qc.h(qubit_pair[0])
                qc.cx(qubit_pair[0], qubit_pair[1])
            qc.barrier()

            return qc
        
        qc = QuantumCircuit(backend.num_qubits,
                            2*len(qubit_pairs)*len(self.tomography_bases),
                            name=f'Bell_State_Tomography')
        
        bit_idx = 0
        for tomography_basis in self.tomography_bases:

            qc = create_bell_states(qc,
                                    qubit_pairs,
                                    bell_state)
            for qubit_pair in qubit_pairs:
        
                apply_rotations_from_list(qc,
                                        tomography_basis,
                                        qubit_pair,
                                        np.arange(bit_idx, bit_idx+len(tomography_basis)))
                bit_idx += 2
            qc.barrier()
        return qc
    
    def _data_analysis(self,
                    qubit_pairs,
                    bell_state):
        
        def create_density_state(tomography_dict):

            density_matrix = np.zeros((4, 4), dtype=complex)
            for label, value in tomography_dict.items():
                pauli_op = Operator(Pauli(label)).data
                density_matrix += (1/4) * value * pauli_op
            return density_matrix
        
        pauli_labels = []
        for i_idx in ['I', 'X', 'Y', 'Z']:
            for j_idx in ['I', 'X', 'Y', 'Z']:
                pauli_labels.append(i_idx+j_idx)

        ideal_bell_states = {
            "phi_plus":  (1/np.sqrt(2)) * np.array([1, 0, 0, 1]),
            "phi_minus": (1/np.sqrt(2)) * np.array([1, 0, 0, -1]),
            "psi_plus":  (1/np.sqrt(2)) * np.array([0, 1, 1, 0]),
            "psi_minus": (1/np.sqrt(2)) * np.array([0, 1, -1, 0]),
        }

        self.bell_state_fidelities = {}
        self.bell_state_fidelities_old = {} # for debugging purposes
        all_tomographies = {}

        for qubit_pair_idx in range(len(qubit_pairs)):

            tomography_dict = {'II': np.float64(1.0)}
            for pauli_term_idx in range(len(self.tomography_bases)):
                pauli_term = self.tomography_bases[pauli_term_idx]
                pauli_term_in_Z = translate_to_Z_basis(pauli_term)
                tomography_dict[pauli_term] = observable_expectation_values_Z_basis([self.ro_corrected_probs_per_n_qubits[qubit_pair_idx][pauli_term_idx]],
                                                                        pauli_term_in_Z)[0]
                
            for sq_pauli_term_idx in range(len(self.single_qubit_terms)):
                sq_pauli_term = self.single_qubit_terms[sq_pauli_term_idx]
                sq_pauli_term_stripped = sq_pauli_term.replace("I", "")
                sq_pauli_term_idx = sq_pauli_term.index(sq_pauli_term_stripped)
                tomography_value = 0
                num_expectation_values = 0
                for pauli_term_idx in range(len(self.tomography_bases)):
                    pauli_term = self.tomography_bases[pauli_term_idx]
                    if sq_pauli_term_stripped == pauli_term[sq_pauli_term_idx]:
                        sq_pauli_term_in_Z = translate_to_Z_basis(sq_pauli_term)
                        expectation_value = observable_expectation_values_Z_basis([self.ro_corrected_probs_per_n_qubits[qubit_pair_idx][pauli_term_idx]],
                                                                                                sq_pauli_term_in_Z)[0]
                        tomography_value += expectation_value
                        num_expectation_values += 1
                tomography_dict[sq_pauli_term] = tomography_value / num_expectation_values

            tomography_dict = {pauli_term: tomography_dict[pauli_term] for pauli_term in pauli_labels}
            tomography_values = [tomography_dict[pauli_term] for pauli_term in tomography_dict.keys()]
            all_tomographies[f"Q{qubit_pairs[qubit_pair_idx]}"] = tomography_dict

            density_matrix_old = create_density_state(tomography_dict)
            density_matrix_real_old = np.real(density_matrix_old)
            density_matrix_imag_old = np.imag(density_matrix_old)            
            fidelity_old = float(np.real(np.vdot(ideal_bell_states[bell_state], density_matrix_old @ ideal_bell_states[bell_state])))
            self.bell_state_fidelities_old[f"Q{qubit_pairs[qubit_pair_idx]} [a.u.]"] = fidelity_old

            density_matrix = self._project_to_density_matrix(tomography_dict,
                                                                 pauli_labels)    
            density_matrix_real = np.real(density_matrix)
            density_matrix_imag = np.imag(density_matrix)      
            fidelity = float(np.real(np.vdot(ideal_bell_states[bell_state], density_matrix @ ideal_bell_states[bell_state])))
            self.bell_state_fidelities[f"Q{qubit_pairs[qubit_pair_idx]} [a.u.]"] = fidelity

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300)
            fig.subplots_adjust(top=0.75, wspace=0.3)
            fig.suptitle(
                f'Bell-state tomography\n{self.backend.name} processor\n'
                f'Qubit pair: Q{qubit_pairs[qubit_pair_idx]} | Fidelity = {100*fidelity:.1f} %\n'
                f'{self.record.date_timestamp}_{self.record.job_timestamp}',
                fontsize=14
            )
            norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            basis_labels = ["00", "01", "10", "11"]

            ax0 = axes[0]
            ax0.bar(range(len(tomography_values)), tomography_values)
            ax0.set_xticks(range(len(tomography_values)))
            ax0.set_xticklabels(pauli_labels, rotation=45)
            ax0.set_xlabel("Observable")
            ax0.set_ylabel("Expectation value")
            ax0.set_ylim(-1.05, 1.05)
            ax0.set_title("Tomography observable values")

            norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            ax1 = axes[1]
            im1 = ax1.imshow(density_matrix_real, cmap="RdBu_r", norm=norm)
            ax1.set_title("Density matrix - real part")
            ax1.set_xticks(range(4))
            ax1.set_yticks(range(4))
            ax1.set_xticklabels(basis_labels)
            ax1.set_yticklabels(basis_labels)
            ax1.set_aspect("equal")
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im1, cax=cax1)

            ax2 = axes[2]
            im2 = ax2.imshow(density_matrix_imag, cmap="RdBu_r", norm=norm)
            ax2.set_title("Density matrix - imaginary part")
            ax2.set_xticks(range(4))
            ax2.set_yticks(range(4))
            ax2.set_xticklabels(basis_labels)
            ax2.set_yticklabels(basis_labels)
            ax2.set_aspect("equal")
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im2, cax=cax2)

            tomography_fig_path = (
                Path(self.record.project_dir)
                / f"tomography_plot_Q{qubit_pairs[qubit_pair_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(tomography_fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Processed data"] = all_tomographies
        self.experiment_data["AllXY deviations"] = self.bell_state_fidelities
        json_file_path = (
            Path(self.record.project_dir)
            / f"tomography_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    def _project_to_density_matrix(self,
                                   tomography_dict: dict,
                                   pauli_labels: list):

        def build_density_matrix(t_params):
            T = np.zeros((4, 4), dtype=complex) # will build lower triangular matrix
            idx = 0
            for i in range(4):
                for j in range(i + 1):
                    if i == j:
                        # real diagonal entries
                        T[i, i] = t_params[idx]
                        idx += 1
                    else:
                        # complex lower triangle
                        real = t_params[idx]
                        imag = t_params[idx + 1]
                        T[i, j] = real + 1j * imag
                        idx += 2
            rho = T.conj().T @ T
            rho /= np.trace(rho) # normalization
            return rho

        def expectation_from_rho(rho, pauli_ops):
            return np.array([
                np.real(np.trace(rho @ P)) for P in pauli_ops
            ])

        def loss(params, measured_expectations, pauli_ops):
            rho = build_density_matrix(params)
            predicted = expectation_from_rho(rho, pauli_ops)
            return np.sum((predicted - measured_expectations)**2)

        measured_expectations = np.array([
            tomography_dict[label] for label in pauli_labels
        ])
        pauli_ops = [
            Operator(Pauli(label)).data for label in pauli_labels
        ]

        init_t_params = np.zeros(16)
        # We start with the identity matrix
        init_t_params[0] = 1.0   # T00
        init_t_params[1] = 1.0   # T11
        init_t_params[3] = 1.0   # T22
        init_t_params[6] = 1.0   # T33

        result = minimize(
            loss,
            init_t_params,
            args=(measured_expectations, pauli_ops),
            method='L-BFGS-B'
        )
        rho_mle = build_density_matrix(result.x)
        return rho_mle

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