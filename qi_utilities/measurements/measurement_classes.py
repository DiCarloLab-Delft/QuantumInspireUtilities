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
        
        fig_all, ax_all = plt.subplots(figsize=(18, 5), dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
                                     for entry in range(len(rotation_angles))]
            params, covariance = curve_fit(cos_func,
                                           rotation_angles,
                                           probabilities_excited)
            a_fit, b_fit, c_fit, d_fit = params
            cosine_fit = cos_func(rotation_angles, a_fit, b_fit, c_fit, d_fit)

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
        
        fig_all, ax_all = plt.subplots(dpi=300)
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
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
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
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
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
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
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
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
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
        
        super().__init__(backend,
                         qubit_list,
                         qc,
                         num_shots,
                         directory)
        
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
            probabilities_excited = [self.ro_corrected_probs_per_qubit[qubit_idx][entry]['1'] \
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
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.ro_corrected_probs_per_qubit[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["AllXY deviations"] = self.allxy_deviations
        json_file_path = (
            Path(self.record.project_dir)
            / f"allxy_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
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