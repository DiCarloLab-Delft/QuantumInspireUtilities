"""
Module implementing various single- and two-qubit characterization
experiments, routinely used in a superconducting qubits lab environment
in order to benchmark a processor's performance and to obtain various
performance metrics, such as a qubit's relaxation time T1 and decoherence
time T2.

Each measurement class contains two main functions, "_quantum_circuit()",
which uniquely defines the experiment quantum circuit(s), and the
"_data_analysis()", which analyzes the experiment output data accordingly.
The quantum circuits created in each measurement class are handled similarly
by a single generic class "BaseMeasurement", which is also responsible for storing
all relevant project record data, such as the raw shots, circuits executed, etc.

The results of each measurement (plots in PNG format and processed data in JSON
format) are stored within the project directory. Most measurement classes also
contain a "load_data()" class method, which loads and re-analyzes the data of
an already executed measurement given its timestamp.

Authors: Marios Samiotis
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, fftfreq
from scipy.stats import sem
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
from lmfit import Model
from uncertainties import ufloat
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
from qi_utilities.utility_functions.circuit_modifiers import apply_rotations_from_list
from qi_utilities.measurements.fitting_functions import (linear_func, exp_decay_func, cos_func, damped_osc_func,
                                                         double_damped_osc_func)
from qi_utilities.utility_functions.raw_data_processing import translate_to_Z_basis, observable_expectation_values_Z_basis
from qi_utilities.measurements.base_classes import BaseMeasurement

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class RabiMeasurement(BaseMeasurement):
    """
    This measurement class implements a Rabi oscillation, which is one of the
    simplest experiments performed with a qubit. The qubit is rotated (driven)
    around the x or y axis of the Bloch sphere, with the probability of measuring
    the qubit in the ground (prob(|0>)) or excited (prob(|1>)) state having a
    sinusoidal response to the driving amplitude.

    Since all single-qubit gate rotations have a fixed duration for any selected backend,
    usually set to 20 ns, the Rabi amplitude is associated with a rotation angle, meaning,
    a larger Rabi amplitude corresponds to a larger single-qubit rotation angle achieved
    within the same single-qubit gate duration.

    This experiment serves as a general check of whether the qubit behaves well
    as a two-level system, since this assumption breaks down when the oscillation of the
    probabilities prob(|0>) or prob(|1>) diverge from the expected sinusoidal response
    with respect to the driving amplitude [1].

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 5, section 5.2.1.
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 rotation_angles: np.array = np.linspace(0, 2*np.pi, num=29),
                 rotation_axis: str = 'x',
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            num_shots (int):
                The number of shots for which the experiment is executed.

            rotation_angles (np.array):
                The different rotation angles for the Rabi oscillation
                expressed in radians [rad].

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        if len(rotation_angles) < 4:
            raise ValueError("Invalid input rotation_angles. Input rotation_angles "
                             "must have at least 4 entries!")
        if rotation_axis not in ['x', 'y']:
            raise ValueError("Invalid input rotation_axis. Available inputs are: "
                             "'x', or 'y'.")
        
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   rotation_angles,
                                   rotation_axis)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points = use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(rotation_angles,
                            rotation_axis)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        
    def _quantum_circuit(self,
                        backend,
                        qubit_list: list,
                        rotation_angles: np.array = np.linspace(0, 2*np.pi, num=29),
                        rotation_axis: str = 'x'):
        """
        This instance method creates the experiment quantum circuit. For each
        rotation angle, a measurement block resets the qubits, applies the
        rotation rx (or ry) and measures all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            rotation_angles (np.array):
                The different rotation angles for the Rabi oscillation
                expressed in radians [rad].

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.
        """

        bit_idx = 0
        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            len(rotation_angles)*len(qubit_list),
                            name=f'Rabi_Oscillation_{rotation_axis}_axis_{backend.name}')

        for step_idx in range(len(rotation_angles)):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                if rotation_axis == 'x':
                    qc.rx(rotation_angles[step_idx], qubit_idx)
                if rotation_axis == 'y':
                    qc.ry(rotation_angles[step_idx], qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()
            
        return qc
    
    def _data_analysis(self,
                       rotation_angles: np.array,
                       rotation_axis: str):
        """
        This instance method fits the excited state population of each qubit with
        the cosine f(x) = a*cos(bx + c) + d (see cos_func in fitting_functions.py),
        keeping the best out of 30 randomly initialized fits based on the
        Bayesian Information Criterion (BIC).

        The fitted excited state population at a rotation angle of pi is stored
        in the attribute 'rabi_amplitudes', with ideal value 1. The plots and the
        experiment data are stored in the project directory.

        Args:
            rotation_angles (np.array):
                The different rotation angles for the Rabi oscillation
                expressed in radians [rad].

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.
        """

        def fit_cosine_function(rotation_angles,
                                probabilities_excited):
            """
            Fits cos_func on the data 30 times with random initial parameters,
            and returns the lmfit result with the lowest BIC.
            """
            
            cos_fit = None
            best_bic = np.inf
            for fit_trial_idx in range(30):
                cos_model = Model(cos_func)
                cos_params = cos_model.make_params(
                    amplitude = np.random.uniform(0.45, 0.55),
                    frequency= np.random.uniform(0.9, 1.1),
                    phase=np.random.uniform(-1.0, 1.0) * np.pi,
                    offset=np.random.uniform(0.45, 0.55)
                )
                cos_params['amplitude'].vary, cos_params['amplitude'].min, cos_params['amplitude'].max = (
                    True, 0.4, 0.6)
                cos_params['frequency'].vary, cos_params['frequency'].min, cos_params['frequency'].max = (
                    True, 0.5, 1.5)
                cos_params['phase'].vary, cos_params['phase'].min, cos_params['phase'].max = (
                    True, -np.pi, np.pi)
                cos_params['offset'].vary, cos_params['offset'].min, cos_params['offset'].max = (
                    True, 0.4, 0.6)
                cos_trial_fit = cos_model.fit(
                    probabilities_excited,
                    cos_params,
                    x=rotation_angles
                )
                if cos_trial_fit.bic < best_bic:
                    best_bic = cos_trial_fit.bic
                    cos_fit = cos_trial_fit
            return cos_fit
        
        self.rabi_amplitudes = {}
        fig_all, ax_all = plt.subplots(figsize=(18, 5), dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.result_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(rotation_angles))]
            
            # Iterate 30 times for best fit
            cos_fit = fit_cosine_function(rotation_angles,
                                          probabilities_excited)
            amplitude_fit, frequency_fit, phase_fit, offset_fit = (
                cos_fit.params['amplitude'].value, cos_fit.params['frequency'].value,
                cos_fit.params['phase'].value, cos_fit.params['offset'].value)
            rotation_angles_fit = np.linspace(rotation_angles[0],
                                              rotation_angles[-1],
                                              num=2001)
            cos_curve = cos_func(rotation_angles_fit, amplitude_fit, frequency_fit,
                                 phase_fit, offset_fit)
            self.rabi_amplitudes[f'{self.qubit_labels[qubit_idx]} [a.u.]'] = cos_func(np.pi, amplitude_fit,
                                                                                      frequency_fit, phase_fit, offset_fit)

            fig, ax = plt.subplots(figsize=(18, 5), dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.plot(rotation_angles,
                        probabilities_excited,
                        label=f'Qubit {self.qubit_labels[qubit_idx]}',
                        alpha=0.6,
                        color = f'C{qubit_idx}',
                        marker='o')
                ax_obj.set_xlabel('Applied rotation')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(
                    f'Rabi oscillation'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.date_timestamp}_{self.job_timestamp}')
        
                labels = []
                for step_idx in range(len(rotation_angles)):
                    angle_in_degrees = (360 / (2 * np.pi)) * rotation_angles[step_idx]
                    if rotation_axis == 'x':
                        step_label = f"rx{round(angle_in_degrees)}"
                    if rotation_axis == 'y':
                        step_label = f"ry{round(angle_in_degrees)}"
                    labels.append(step_label)
                label_locs = rotation_angles
                ax_obj.set_xticks(label_locs)
                ax_obj.set_xticklabels(labels, rotation=65)
                ax_obj.set_ylim(-0.05, 1.05)
                ax_obj.grid(True)
            ax.plot(rotation_angles_fit,
                    cos_curve,
                    alpha=0.7,
                    color = 'red',
                    label = 'Fit: ' + r'$f(x) = a\cdot\cos (bx + c) + d$')
            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=r'$a=$' + f'{amplitude_fit:.2f}')
            ax.plot([], [], ' ', label=r'$b=$' + f'{frequency_fit:.2f}')
            ax.plot([], [], ' ', label=r'$c=$' + f'{(phase_fit/np.pi):.2f}' + r'$\cdot \pi$')
            ax.plot([], [], ' ', label=r'$d=$' + f'{offset_fit:.2f}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
            rabi_fig_path = (
                Path(self.project_dir)
                / f"rabi_plot_{self.qubit_labels[qubit_idx]}_{self.date_timestamp}_{self.job_timestamp}.png"
            )
            fig.savefig(rabi_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        rabi_all_fig_path = (
            Path(self.project_dir)
            / f"rabi_plot_ALL_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig_all.savefig(rabi_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.date_timestamp}_{self.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["Rotation angles [rad]"] = rotation_angles
        self.experiment_data["Rotation axis"] = rotation_axis
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["Rabi amplitudes"] = self.rabi_amplitudes
        json_file_path = (
            Path(self.project_dir)
            / f"rabi_data_{self.date_timestamp}_{self.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        rotation_angles = json_data['Rotation angles [rad]']
        rotation_axis = json_data['Rotation axis']
        cls.qubit_list = json_data['Qubit list']
        cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls,
                           rotation_angles,
                           rotation_axis)
        return json_data

class T1_Measurement(BaseMeasurement):
    """
    This measurement class implements a T1 measurement, used for obtaining the
    energy relaxation time T1 of a qubit. The qubit is prepared in its excited
    state |1> with a pi pulse, left idle for a variable time t, and then measured.

    Due to energy relaxation, the excited state population decays exponentially
    towards the ground state |0>, as prob(|1>) = a*exp(-t/T1) + b [1, 2]. The
    relaxation time T1 also sets an upper limit to the decoherence time, T2 <= 2*T1 [1].

    The measurement times are expected to be equidistant starting from t = 0,
    while the total measurement time should be a few times the expected T1
    (by default in [3], 4*T1) for a reliable fit. On the Tuna backends, the
    idle times are applied in integer multiples of the 20 ns cycle time.

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 2, section 2.1.5.
    [2] Control for Programmable Superconducting Quantum Systems,
        Michiel Adriaan Rol, PhD Thesis, Delft University of Technology (2020)
        https://doi.org/10.4233/uuid:0a2ba212-f6bf-4c64-8f3d-b707f1e44953
        Chapter 3, section 3.3.1.
    [3] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Transmon.measure_T1()
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 measurement_times: np.array = np.linspace(0, 150e-6, num=41),
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            num_shots (int):
                The number of shots for which the experiment is executed.

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        
        qc = self._quantum_circuit(backend=backend,
                                   qubit_list=qubit_list,
                                   measurement_times=measurement_times)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points=use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(measurement_times)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        

    def _quantum_circuit(self,
                        backend,
                        qubit_list: list,
                        measurement_times: np.array = np.linspace(0, 150e-6, num=41)):
        """
        This instance method creates the experiment quantum circuit. For each
        measurement time, a measurement block resets the qubits, prepares them
        in |1> with an X gate, idles for the corresponding time and measures
        all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        """

        cycle_time = 20e-9 # cycle time limitation imposed by the Tuna backend internal software
        total_time = measurement_times[-1]
        num_points = len(measurement_times)
        dt = total_time / (num_points - 1)
        bit_idx = 0

        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            num_points*len(qubit_list),
                            name=f'T1_{backend.name}')
        
        for time_idx in range(num_points):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.x(qubit_idx) # qubit initialization to the |1> state
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.delay(duration = int(np.round(time_idx * dt / cycle_time)),
                        qarg = qubit_idx) # delay in units of cycle_time
            qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()

        return qc
    
    def _data_analysis(self,
                       measurement_times: np.array):
        """
        This instance method fits the excited state population of each qubit with
        the exponential decay f(t) = a*exp(-t/tau) + b (see exp_decay_func in
        fitting_functions.py). The fitted decay times are stored in the attribute
        'T1_values' in units of [us], while the plots and the experiment data are
        stored in the project directory.

        Args:
            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        """

        self.T1_values = {}
        
        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.result_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            params, covariance = curve_fit(exp_decay_func,
                                           measurement_times,
                                           probabilities_excited,
                                           bounds=(
                                                [0, 0, 0],   # lower bounds
                                                [np.inf, 1.0, 0.5]  # upper bounds
                                            )
                                           )
            stderr = np.sqrt(np.diag(covariance))
            tau_fit, amplitude_fit, offset_fit = params
            measurement_times_fit = np.linspace(measurement_times[0],
                                                measurement_times[-1],
                                                num = 101)
            exponential_fit = exp_decay_func(measurement_times_fit, tau_fit, amplitude_fit, offset_fit)
            self.T1_values[f'{self.qubit_labels[qubit_idx]} [us]'] = np.round(1e6 * tau_fit, 1)

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.plot(1e6*measurement_times,
                            probabilities_excited,
                            label=f'{self.qubit_labels[qubit_idx]}: T1 = ({1e6 * tau_fit:.1f}'+r'$\pm$'+f'{1e6 * stderr[0]:.1f}) μs',
                            alpha=0.6,
                            color = f'C{qubit_idx}',
                            marker='o')
                ax_obj.set_xlabel('Time (μs)')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(
                    f'T1 measurement'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.date_timestamp}_{self.job_timestamp}')
                ax_obj.set_ylim(-0.05, 1.05)
            ax.plot(1e6*measurement_times_fit,
                    exponential_fit,
                    alpha=0.7,
                    color = 'red',
                    label='Fit: ' + r'$f(t) = a\cdot\exp (-t/\tau) + b$')
            ax.legend()
            T1_fig_path = (
                Path(self.project_dir)
                / f"T1_plot_{self.qubit_labels[qubit_idx]}_{self.date_timestamp}_{self.job_timestamp}.png"
            )
            fig.savefig(T1_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        T1_all_fig_path = (
            Path(self.project_dir)
            / f"T1_plot_ALL_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig_all.savefig(T1_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.date_timestamp}_{self.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T1 values"] = self.T1_values
        json_file_path = (
            Path(self.project_dir)
            / f"T1_data_{self.date_timestamp}_{self.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        measurement_times = np.array(json_data['Measurement times [s]'])
        cls.qubit_list = json_data['Qubit list']
        cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls,
                           measurement_times)
        return json_data

class T2_RamseyMeasurement(BaseMeasurement):
    """
    This measurement class implements a Ramsey experiment, used for obtaining
    the dephasing time T2* of a qubit, as well as the detuning of the qubit
    from its drive frequency. The qubit is brought to the equator of the Bloch
    sphere with a pi/2 pulse, left to evolve freely for a variable time t, and
    rotated back with a second pi/2 pulse prior to measurement [1].

    A qubit detuning acts as a continuous Z rotation during the free evolution,
    while dephasing decays the oscillation, so that the excited state population
    follows the damped oscillation prob(|1>) = a*exp(-t/T2*)*cos(2*pi*f*t + phi) + b.

    In order to distinguish the decay from the oscillation, an artificial
    detuning f_art is imposed by a virtual Z rotation of angle -2*pi*f_art*t
    prior to the second pi/2 pulse, which is equivalent to sweeping the phase of
    the second pi/2 pulse [1]. Here f_art = 5 / measurement_times[-1], giving
    five oscillations within the measurement window [2]. The residual qubit
    detuning is the difference between the fitted frequency f and f_art [1].

    The measurement times are expected to be equidistant starting from t = 0.
    On the Tuna backends, the idle times are applied in integer multiples of
    the 20 ns cycle time.

    The data are additionally fitted with a double damped oscillation and
    analyzed with an FFT, in order to detect a double beating (two oscillation
    frequencies, e.g. due to a qubit frequency switching between two values).
    If a double beating is detected, based on the BIC difference of the two fits
    or the number of FFT peaks, the experiment is re-launched with two adjusted
    artificial detunings f' = |f_art - (f1+f2)/2| and f' = f_art + (f1+f2)/2,
    and T2* is extracted from the configuration in which a single damped
    oscillation describes the data best.

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 5, section 5.2.2.
    [2] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Transmon.measure_ramsey()
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 measurement_times: np.array,
                 num_shots: int,
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].

            num_shots (int):
                The number of shots for which the experiment is executed.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        # Here 5 is set for five periods
        self.artificial_detunings = 5 / measurement_times[-1] * np.ones(len(qubit_list))
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   measurement_times,
                                   self.artificial_detunings)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points=use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(measurement_times)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        

    def _quantum_circuit(self,
                         backend,
                         qubit_list: list,
                         measurement_times: np.array,
                         artificial_detunings: list[float]):
        """
        This instance method creates the experiment quantum circuit. For each
        measurement time, a measurement block resets the qubits, applies an
        rx(pi/2) pulse, idles for the corresponding time, applies the virtual
        Z rotation of the artificial detuning and a second rx(pi/2) pulse, and
        measures all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].

            artificial_detunings (list[float]):
                The artificial detuning of each qubit expressed in units of [Hz].
        """

        cycle_time = 20e-9 # cycle time limitation imposed by the Tuna backend internal software
        total_time = measurement_times[-1]
        num_points = len(measurement_times)
        dt = total_time / (num_points - 1)
        bit_idx = 0

        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            num_points*len(qubit_list),
                            name=f'T2_Ramsey_{backend.name}')

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
                qc.delay(duration = int(np.round(time_idx * dt / cycle_time)),
                        qarg = qubit_idx)
            qc.barrier()

            # adding the artificial detuning which rotates the qubit state
            for list_idx, qubit_idx in enumerate(qubit_list):
                qc.rz(- (2*np.pi) * artificial_detunings[list_idx] * (time_idx*dt),
                    qubit_idx)
            qc.barrier()

            # rotating qubit state
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
        """
        This instance method fits the excited state population of each qubit with
        both a single and a double damped oscillation (see damped_osc_func and
        double_damped_osc_func in fitting_functions.py), keeping the best out of
        multiple randomly initialized fits based on the Bayesian Information
        Criterion (BIC), and computes the FFT of the data.

        A double beating is detected when the relative BIC difference of the two
        fits exceeds 30%, or when more than one peak is found in the FFT. In that
        case, the experiment is re-launched with the two adjusted artificial
        detunings (see the class description), and the decay time of the
        configuration with the smallest relative BIC difference is kept.

        The fitted decay times are stored in the attribute 'T2_ramsey_values' in units
        of [us], while the plots and the experiment data are stored in the project
        directory.

        Args:
            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        """
        
        def fit_dumped_osc_function(measurement_times,
                                    probabilities_excited,
                                    artificial_detuning,
                                    damped_osc_freq_lim: int):
            """
            Fits damped_osc_func on the data 50 times with random initial parameters,
            and returns the lmfit result with the lowest BIC. The fitted frequency
            is bounded in [artificial_detuning/4, damped_osc_freq_lim*artificial_detuning].
            """
            
            damped_oscillation_fit = None
            best_bic = np.inf
            for fit_trial_idx in range(50):
                damped_osc_model = Model(damped_osc_func)
                damped_osc_params = damped_osc_model.make_params(
                    tau = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency= np.random.uniform(0.5, damped_osc_freq_lim/4) * artificial_detuning,
                    phase=np.random.uniform(0.0, 0.5) * np.pi,
                    amplitude=np.random.uniform(0.4, 0.6),
                    offset=np.random.uniform(0.4, 0.6)
                )
                damped_osc_params['tau'].vary, damped_osc_params['tau'].min, damped_osc_params['tau'].max = (
                    True, 0, 5*measurement_times[-1])
                damped_osc_params['frequency'].vary, damped_osc_params['frequency'].min, damped_osc_params['frequency'].max = (
                    True, artificial_detuning/4, damped_osc_freq_lim*artificial_detuning)
                damped_osc_params['phase'].vary, damped_osc_params['phase'].min, damped_osc_params['phase'].max = (
                    True, -np.pi, np.pi)
                damped_osc_params['amplitude'].vary, damped_osc_params['amplitude'].min, damped_osc_params['amplitude'].max = (
                    True, 0.4, 0.6)
                damped_osc_params['offset'].vary, damped_osc_params['offset'].min, damped_osc_params['offset'].max = (
                    True, 0.1, 0.9)
                damped_oscillation_trial_fit = damped_osc_model.fit(
                    probabilities_excited,
                    damped_osc_params,
                    t=measurement_times
                )
                if damped_oscillation_trial_fit.bic < best_bic:
                    best_bic = damped_oscillation_trial_fit.bic
                    damped_oscillation_fit = damped_oscillation_trial_fit
            return damped_oscillation_fit
        
        def fit_double_dumped_osc_function(measurement_times,
                                           probabilities_excited,
                                           artificial_detuning):
            """
            Fits double_damped_osc_func on the data 30 times with random initial
            parameters, and returns the lmfit result with the lowest BIC.
            """

            double_damped_oscillation_fit = None
            best_bic = np.inf
            for fit_trial_idx in range(30):
                double_damped_osc_model = Model(double_damped_osc_func)
                double_damped_osc_params = double_damped_osc_model.make_params(
                    tau_1 = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency_1 = np.random.uniform(0.5, 1.5)*artificial_detuning,
                    phase_1=0.0,
                    amplitude_1=0.5*(max(probabilities_excited) - min(probabilities_excited)),

                    tau_2 = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency_2 = np.random.uniform(0.5, 1.5)*artificial_detuning,
                    phase_2=0.0,
                    amplitude_2=0.5*(max(probabilities_excited) - min(probabilities_excited)),

                    offset=0.5
                )
                double_damped_osc_params['tau_1'].vary, double_damped_osc_params['tau_1'].min, double_damped_osc_params['tau_1'].max = (
                    True, 0.05*measurement_times[-1], 5*measurement_times[-1])
                double_damped_osc_params['frequency_1'].vary, double_damped_osc_params['frequency_1'].min, double_damped_osc_params['frequency_1'].max = (
                    True, artificial_detuning/4, 4*artificial_detuning)
                double_damped_osc_params['phase_1'].vary, double_damped_osc_params['phase_1'].min, double_damped_osc_params['phase_1'].max = (
                    True, -np.pi, np.pi)
                double_damped_osc_params['amplitude_1'].vary, double_damped_osc_params['amplitude_1'].min, double_damped_osc_params['amplitude_1'].max = (
                    True, 0.1, 0.9)
                
                double_damped_osc_params['tau_2'].vary, double_damped_osc_params['tau_2'].min, double_damped_osc_params['tau_2'].max = (
                    True, 0.05*measurement_times[-1], 5*measurement_times[-1])
                double_damped_osc_params['frequency_2'].vary, double_damped_osc_params['frequency_2'].min, double_damped_osc_params['frequency_2'].max = (
                    True, artificial_detuning/4, 4*artificial_detuning)
                double_damped_osc_params['phase_2'].vary, double_damped_osc_params['phase_2'].min, double_damped_osc_params['phase_2'].max = (
                    True, -np.pi, np.pi)
                double_damped_osc_params['amplitude_2'].vary, double_damped_osc_params['amplitude_2'].min, double_damped_osc_params['amplitude_2'].max = (
                    True, 0.1, 0.9)
                double_damped_osc_params['offset'].vary, double_damped_osc_params['offset'].min, double_damped_osc_params['offset'].max = (
                    True, 0.48, 0.52)
                double_damped_oscillation_trial_fit = double_damped_osc_model.fit(
                    probabilities_excited,
                    double_damped_osc_params,
                    t=measurement_times
                )
                if double_damped_oscillation_trial_fit.bic < best_bic:
                    best_bic = double_damped_oscillation_trial_fit.bic
                    double_damped_oscillation_fit = double_damped_oscillation_trial_fit
            return double_damped_oscillation_fit
        
        self.T2_ramsey_values = {}
        bic_diff_tolerance = 0.30
        result_probs_per_n_qubits = self.result_probs_per_n_qubits
        probabilities_excited_per_n_qubits = []
        artificial_detunings = {'original': self.artificial_detunings.copy(),
                                'minus': self.artificial_detunings.copy(),
                                'plus': self.artificial_detunings.copy()}
        damped_osc_curves = {'original': [], 'minus': [], 'plus': []}
        double_damped_osc_curves = {'original': [], 'minus': [], 'plus': []}
        damped_osc_fits = {'minus': [], 'plus': []}
        double_damped_osc_fits = {'minus': [], 'plus': []}
        bic_avgs = {'minus': [], 'plus': []}
        bic_diffs = {'minus': [], 'plus': []}
        tau_fits = {'original': [], 'minus': [], 'plus': []}
        tau_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        tau_1_fits = {'original': [], 'minus': [], 'plus': []}
        tau_1_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        tau_2_fits = {'original': [], 'minus': [], 'plus': []}
        tau_2_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        frequency_1_fits = {'original': [], 'minus': [], 'plus': []}
        frequency_2_fits = {'original': [], 'minus': [], 'plus': []}
        fit_bics = {'single_beating': [], 'double_beating': [], 'single_minus': [], 'single_plus': []}
        xf_data_per_n_qubits = []
        fft_data_amplitude_per_n_qubits = []
        xf_fit_per_n_qubits = []
        fft_fit_amplitude_per_n_qubits = []
        peak_freqs_per_n_qubits = []
        peak_amps_per_n_qubits = []
        fitted_double_osc_pass = [False for qubit_idx in range(len(self.qubit_list))]

        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [result_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            probabilities_excited_per_n_qubits.append(probabilities_excited)

            # Iterate 30 times for best fit
            damped_oscillation_fit = fit_dumped_osc_function(measurement_times,
                                                             probabilities_excited,
                                                             artificial_detunings['original'][qubit_idx],
                                                             damped_osc_freq_lim=10)
            tau_fit, frequency_fit, phase_fit, amplitude_fit, offset_fit = (damped_oscillation_fit.params['tau'].value,
                damped_oscillation_fit.params['frequency'].value, damped_oscillation_fit.params['phase'].value,
                damped_oscillation_fit.params['amplitude'].value, damped_oscillation_fit.params['offset'].value)
            tau_stderr = damped_oscillation_fit.params['tau'].stderr
            measurement_times_fit = np.linspace(measurement_times[0],
                                                measurement_times[-1],
                                                num = 2001)
            damped_oscillation_curve = damped_osc_func(measurement_times_fit, tau_fit, frequency_fit,
                                                       phase_fit, amplitude_fit, offset_fit)
            tau_fits['original'].append(tau_fit)
            tau_fits_stderr['original'].append(tau_stderr)
            damped_osc_curves['original'].append(damped_oscillation_curve)
            self.T2_ramsey_values[f'{self.qubit_labels[qubit_idx]} [us]'] = np.round(1e6 * tau_fit, 1)

            # Iterate 30 times for best fit
            double_damped_oscillation_fit = fit_double_dumped_osc_function(measurement_times,
                                                                           probabilities_excited,
                                                                           artificial_detunings['original'][qubit_idx])
            (tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
             tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
             offset_fit) = (
                            double_damped_oscillation_fit.params['tau_1'].value, double_damped_oscillation_fit.params['frequency_1'].value, 
                            double_damped_oscillation_fit.params['phase_1'].value, double_damped_oscillation_fit.params['amplitude_1'].value,
                            double_damped_oscillation_fit.params['tau_2'].value, double_damped_oscillation_fit.params['frequency_2'].value, 
                            double_damped_oscillation_fit.params['phase_2'].value, double_damped_oscillation_fit.params['amplitude_2'].value,
                            damped_oscillation_fit.params['offset'].value)
            (tau_1_stderr, tau_2_stderr) = (double_damped_oscillation_fit.params['tau_1'].stderr,
                                            double_damped_oscillation_fit.params['tau_2'].stderr)
            double_damped_oscillation_curve = double_damped_osc_func(measurement_times_fit,
                                                                     tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
                                                                     tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
                                                                     offset_fit)
            tau_1_fits['original'].append(tau_1_fit)
            tau_2_fits['original'].append(tau_2_fit)
            tau_1_fits_stderr['original'].append(tau_1_stderr)
            tau_2_fits_stderr['original'].append(tau_2_stderr)
            frequency_1_fits['original'].append(frequency_1_fit)
            frequency_2_fits['original'].append(frequency_2_fit)
            double_damped_osc_curves['original'].append(double_damped_oscillation_curve)

            pad_N = 5000
            N_data = len(measurement_times)
            T_data = measurement_times[1] - measurement_times[0]
            signal_data = probabilities_excited - np.mean(probabilities_excited) # remove DC component for FFT
            window_data = np.hanning(N_data)
            signal_data = signal_data * window_data # apply window to data
            yf_data = fft(signal_data, n=pad_N)
            xf_data = fftfreq(pad_N, T_data)[:pad_N//2]
            fft_data_amplitude = 2.0 * np.abs(yf_data[:pad_N//2]) / np.sum(window_data)
            xf_data_per_n_qubits.append(xf_data)
            fft_data_amplitude_per_n_qubits.append(fft_data_amplitude)

            signal_fit = double_damped_osc_func(measurement_times,
                                                tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
                                                tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
                                                offset_fit)
            signal_fit -= np.mean(signal_fit)
            window_fit = np.hanning(N_data)
            signal_fit = signal_fit * window_fit # apply window to data
            yf_fit = fft(signal_fit, n=pad_N)
            xf_fit = fftfreq(pad_N, T_data)[:pad_N//2]
            fft_fit_amplitude = 2.0 * np.abs(yf_fit[:pad_N//2]) / np.sum(window_fit)
            xf_fit_per_n_qubits.append(xf_fit)
            fft_fit_amplitude_per_n_qubits.append(fft_fit_amplitude)


            peaks, properties = find_peaks(fft_data_amplitude,
                                           prominence=0.1 * np.max(fft_data_amplitude))
            peak_freqs = xf_data[peaks]
            peak_amps = fft_data_amplitude[peaks]
            peak_freqs_per_n_qubits.append(peak_freqs)
            peak_amps_per_n_qubits.append(peak_amps)

            fit_bics['single_beating'].append(damped_oscillation_fit.bic)
            fit_bics['double_beating'].append(double_damped_oscillation_fit.bic)
            bic_avg = (damped_oscillation_fit.bic+double_damped_oscillation_fit.bic) /2
            bic_diff = np.abs(damped_oscillation_fit.bic-double_damped_oscillation_fit.bic) / np.abs(bic_avg)
            if bic_diff > bic_diff_tolerance or len(peaks)>1: # if bic difference is larger than bic_diff_tolerance
                                                # or more than 1 peaks have been detected in the FFT
                fitted_double_osc_pass[qubit_idx] = True
                original_detuning = artificial_detunings['original'][qubit_idx].copy()
                f1 = frequency_1_fit
                f2 = frequency_2_fit
                artificial_detunings['minus'][qubit_idx] = np.abs(original_detuning - (f1+f2)/2)
                artificial_detunings['plus'][qubit_idx] = original_detuning + (f1+f2)/2

        if any(fitted_double_osc_pass):
            double_beating_labels = []
            for qubit_idx in range(len(self.qubit_list)):
                if fitted_double_osc_pass[qubit_idx] == True:
                    double_beating_labels.append(self.qubit_labels[qubit_idx])
            timestamp_now = datetime.now().replace(microsecond=0)
            print(f"[{timestamp_now}] "
                "Detected double beating for qubits "
                f"{double_beating_labels}.")
            qc_minus = self._quantum_circuit(backend=self.backend,
                                        qubit_list=self.qubit_list,
                                        measurement_times=measurement_times,
                                        artificial_detunings=artificial_detunings['minus'])
            qc_plus = self._quantum_circuit(backend=self.backend,
                                        qubit_list=self.qubit_list,
                                        measurement_times=measurement_times,
                                        artificial_detunings=artificial_detunings['plus'])
            print(f"[{timestamp_now}] "
                "Re-launching the measurement routine with adjusted artificial detunings to get rid of the double beating(s) ... ")
            super().__init__(backend=self.backend,
                                qubit_list=self.qubit_list,
                                qc=[qc_minus, qc_plus],
                                num_shots=self.num_shots,
                                store_record=True)   
                               
        for qubit_idx in range(len(self.qubit_list)):    
            if fitted_double_osc_pass[qubit_idx] == True:
                config = ['minus', 'plus']
                multi_probabilities_excited = {}
                for idx in range(2):

                    probabilities_excited_new = [self.result_multi_probs_per_n_qubits[idx][qubit_idx][entry]['1'] \
                                                for entry in range(len(measurement_times))]
                    multi_probabilities_excited[config[idx]] = probabilities_excited_new
                    damped_osc_fits[config[idx]] = fit_dumped_osc_function(measurement_times,
                                                                          probabilities_excited_new,
                                                                          artificial_detunings[config[idx]][qubit_idx],
                                                                          damped_osc_freq_lim=17)
                    double_damped_osc_fits[config[idx]] = fit_double_dumped_osc_function(measurement_times,
                                                                                         probabilities_excited_new,
                                                                                         artificial_detunings[config[idx]][qubit_idx])
                    bic_avgs[config[idx]] = (damped_osc_fits[config[idx]].bic+double_damped_osc_fits[config[idx]].bic) /2
                    bic_diffs[config[idx]] = np.abs(damped_osc_fits[config[idx]].bic-double_damped_osc_fits[config[idx]].bic) \
                                            / np.abs(bic_avgs[config[idx]])

                if bic_diffs['minus'] < bic_diffs['plus']:
                    winning_config = 'minus'
                else: winning_config = 'plus'
                tau_fits[winning_config], frequency_fits, phase_fit_new, amplitude_fit_new, offset_fit_new = (
                damped_osc_fits[winning_config].params['tau'].value, damped_osc_fits[winning_config].params['frequency'].value,
                damped_osc_fits[winning_config].params['phase'].value, damped_osc_fits[winning_config].params['amplitude'].value,
                damped_osc_fits[winning_config].params['offset'].value)
                tau_stderr = damped_osc_fits[winning_config].params['tau'].stderr
                tau_fits_stderr[winning_config].append(tau_stderr)
                damped_osc_curve = damped_osc_func(measurement_times_fit, tau_fits[winning_config], frequency_fits,
                                                    phase_fit_new, amplitude_fit_new, offset_fit_new)
                self.T2_ramsey_values[f'{self.qubit_labels[qubit_idx]} [us]'] = np.round(1e6 * tau_fits[winning_config], 1)

                fig, axes = plt.subplot_mosaic(
                            [
                                ["A", "B"],
                                ["C", "C"]
                            ],
                            figsize=(12, 12),
                            dpi=300
                        )
                ax = axes["A"]
                ax1 = axes["B"]
                ax2 = axes["C"]
                fig.suptitle(
                    f'T2 Ramsey measurement'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.record.date_timestamp}_{self.record.job_timestamp}',
                    fontsize=14
                )
            else:
                fig, ax = plt.subplots(dpi=300)
            ax.plot(1e6*measurement_times,
                    probabilities_excited_per_n_qubits[qubit_idx],
                    label=f'{self.qubit_labels[qubit_idx]}: T2* = {1e6 * tau_fits['original'][qubit_idx]:.1f} μs',
                    alpha=0.6,
                    color = f'C{qubit_idx}',
                    marker='o')
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel(r'Population $|1\rangle$')
            ax.set_ylim(-0.05, 1.05)
            ax.plot(1e6*measurement_times_fit,
                        damped_osc_curves['original'][qubit_idx],
                        alpha=0.7,
                        color = 'red',
                        label = 'Single oscillation fit')
            if fitted_double_osc_pass[qubit_idx] == True:
                ax.plot(1e6*measurement_times_fit,
                            double_damped_osc_curves['original'][qubit_idx],
                            alpha=0.7,
                            color = 'green',
                            label = 'Double oscillation fit')
                ax.plot([], [], ' ', label=' ')
                ax.plot([], [], ' ', label=f'Artificial detuning ' + r'$f = $' + f'{artificial_detunings['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$f_1 = $' + f'{frequency_1_fits['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$f_2 = $' + f'{frequency_2_fits['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$T_2^{* (1)} = $' + f'{1e6 * tau_1_fits['original'][qubit_idx]:.1f} μs')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$T_2^{* (2)} = $' + f'{1e6 * tau_2_fits['original'][qubit_idx]:.1f} μs')
                ax.set_title("Initial measurement")
                ax.legend(loc='center left', bbox_to_anchor=(-0.9, 0.75))

                ax1.plot(xf_data_per_n_qubits[qubit_idx],
                         fft_data_amplitude_per_n_qubits[qubit_idx],
                        label=f'{self.qubit_labels[qubit_idx]} data',
                        alpha=0.8,
                        color=f'C{qubit_idx}',
                        marker='o',
                        markersize=0.5)
                ax1.plot(xf_fit_per_n_qubits[qubit_idx],
                         fft_fit_amplitude_per_n_qubits[qubit_idx],
                        label=f'{self.qubit_labels[qubit_idx]} double osc. fit',
                        alpha=0.6,
                        color='green')
                ax1.axvline(frequency_1_fits['original'][qubit_idx],
                        label='Double osc. fit ' + r'$f_1$',
                        linestyle = '--',
                        linewidth = 1.2,
                        alpha = 0.7,
                        color='black')
                ax1.axvline(frequency_2_fits['original'][qubit_idx],
                        label='Double osc. fit ' + r'$f_2$',
                        linestyle='--',
                        linewidth = 1.2,
                        alpha = 0.9,
                        color='orange')
                ax1.scatter(
                            peak_freqs_per_n_qubits[qubit_idx],
                            peak_amps_per_n_qubits[qubit_idx],
                            marker='*',
                            s=150,
                            color='orange',
                            edgecolors='black',
                            linewidths=1.5,
                            label=f'Detected peaks (on {self.qubit_labels[qubit_idx]} data)',
                            zorder=10
                        )
                ax1.set_title("FFT of initial measurement data")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Amplitude")
                ax1.set_xlim(0, xf_data[-1]) # restrict frequency values to those of the data
                ax1.legend(loc='center right', bbox_to_anchor=(1.65, 0.85))

                markers = {'minus': 's', 'plus': '^'}
                colors = {'minus': 'darkred', 'plus': 'purple'}
                sign = {'minus': '-', 'plus': '+'}
                ax2.plot(1e6*measurement_times,
                        multi_probabilities_excited[winning_config],
                        label=f'{self.qubit_labels[qubit_idx]}: T2* = {1e6 * tau_fits[winning_config]:.1f} μs',
                        alpha=0.6,
                        color = f'C{qubit_idx}',
                        marker=markers[winning_config])
                ax2.plot(1e6*measurement_times_fit,
                        damped_osc_curve,
                        alpha=0.7,
                        color = colors[winning_config],
                        label = 'Single oscillation fit')
                ax2.plot([], [], ' ', label=' ')
                ax2.plot([], [], ' ', label=f'Artificial detuning: ' + r"$f' = $" + f'{artificial_detunings[winning_config][qubit_idx]:.1f} Hz')
                ax2.plot([], [], ' ', label=' ')
                ax2.plot([], [], ' ', label=f'Artificial detuning adjustment:')
                ax2.plot([], [], ' ', label=r"$f' = f$"+f'{sign[winning_config]}'+r'$(f_1+f_2)/2$')
                ax2.set_title('Final measurement with adjusted artificial detuning')
                ax2.set_xlabel('Time (μs)')
                ax2.set_ylabel(r'Population $|1\rangle$')
                ax2.set_ylim(-0.05, 1.05)
                ax2.legend(loc='center left', bbox_to_anchor=(-0.43, 0.8))
            else:
                ax.set_title(
                f'T2 Ramsey measurement'
                f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                f'\n{self.record.date_timestamp}_{self.record.job_timestamp}')
                ax.legend()

            T2_ramsey_fig_path = (
                Path(self.record.project_dir)
                / f"T2_Ramsey_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(T2_ramsey_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Artificial detunings [Hz]"] = artificial_detunings['original']
        self.experiment_data["Artificial detunings minus [Hz]"] = artificial_detunings['minus']
        self.experiment_data["Artificial detunings plus [Hz]"] = artificial_detunings['plus']
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T2 Ramsey values"] = self.T2_ramsey_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T2_Ramsey_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)


class T2_EchoMeasurement(BaseMeasurement):
    """
    This measurement class implements a Hahn echo experiment, used for obtaining
    the echo dephasing time T2echo of a qubit. The sequence is identical to the
    Ramsey experiment (see T2_RamseyMeasurement), with an additional pi pulse
    applied exactly in the middle of the free evolution time.

    The pi pulse reverses the direction of the qubit evolution, making the qubit
    insensitive to low-frequency noise. Since the noise spectrum of superconducting
    qubits is usually skewed towards low frequencies, typically T2echo > T2* [1].

    Similarly to the Ramsey experiment, an artificial detuning is imposed by a
    virtual Z rotation prior to the final pi/2 pulse, and the same double beating
    detection and re-launch procedure is followed.

    The measurement times are expected to be equidistant starting from t = 0.
    On the Tuna backends, each half of the free evolution time is applied in
    integer multiples of the 20 ns cycle time, therefore the measurement times
    should be integer multiples of 40 ns.

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 5, section 5.2.2 (footnote on the Hahn echo).
    [2] Control for Programmable Superconducting Quantum Systems,
        Michiel Adriaan Rol, PhD Thesis, Delft University of Technology (2020)
        https://doi.org/10.4233/uuid:0a2ba212-f6bf-4c64-8f3d-b707f1e44953
        Chapter 3, section 3.3 (Table 3.1).
    [3] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Transmon.measure_echo()
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 measurement_times: np.array,
                 num_shots: int,
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            measurement_times (np.array):
                The total free evolution times (between the two pi/2 pulses)
                expressed in units of seconds [s].

            num_shots (int):
                The number of shots for which the experiment is executed.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        # Here 5 is set for five periods
        self.artificial_detunings = 5 / measurement_times[-1] * np.ones(len(qubit_list))
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   measurement_times,
                                   self.artificial_detunings)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points=use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(measurement_times)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        

    def _quantum_circuit(self,
                         backend,
                         qubit_list: list,
                         measurement_times: np.array,
                         artificial_detunings: list[float]):
        """
        This instance method creates the experiment quantum circuit. For each
        measurement time, a measurement block resets the qubits, applies an
        rx(pi/2) pulse, idles for half the measurement time, applies an rx(pi)
        echo pulse, idles for the second half, applies the virtual Z rotation of
        the artificial detuning and a second rx(pi/2) pulse, and measures all
        qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            measurement_times (np.array):
                The total free evolution times (between the two pi/2 pulses)
                expressed in units of seconds [s].

            artificial_detunings (list[float]):
                The artificial detuning of each qubit expressed in units of [Hz].
        """

        cycle_time = 20e-9 # cycle time limitation imposed by the Tuna backend internal software
        total_time = measurement_times[-1]
        num_points = len(measurement_times)
        dt = total_time / (num_points - 1)
        bit_idx = 0

        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            num_points*len(qubit_list),
                            name=f'T2_Echo_{backend.name}')

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
                qc.delay(duration = int(np.round(time_idx * (dt/2) / cycle_time)),
                        qarg = qubit_idx)
            qc.barrier()
            # applying the echo pulse
            for qubit_idx in qubit_list:
                qc.rx(np.pi, qubit_idx)
            qc.barrier()
            # delay/2 in units of cycle_time
            for qubit_idx in qubit_list:
                qc.delay(duration = int(np.round(time_idx * (dt/2) / cycle_time)),
                        qarg = qubit_idx)
            qc.barrier()

            # adding the artificial detuning which rotates the qubit state
            for list_idx, qubit_idx in enumerate(qubit_list):
                qc.rz(- (2*np.pi) * artificial_detunings[list_idx] * (time_idx*dt),
                    qubit_idx)
            qc.barrier()

            # rotating qubit state
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
        """
        This instance method fits the excited state population of each qubit with
        both a single and a double damped oscillation (see damped_osc_func and
        double_damped_osc_func in fitting_functions.py), keeping the best out of
        multiple randomly initialized fits based on the Bayesian Information
        Criterion (BIC), and computes the FFT of the data.

        A double beating is detected when the relative BIC difference of the two
        fits exceeds 30%, or when more than one peak is found in the FFT. In that
        case, the experiment is re-launched with the two adjusted artificial
        detunings (see the class description), and the decay time of the
        configuration with the smallest relative BIC difference is kept.

        The fitted decay times are stored in the attribute 'T2_echo_values' in units
        of [us], while the plots and the experiment data are stored in the project
        directory.

        Args:
            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].
        """
        
        def fit_dumped_osc_function(measurement_times,
                                    probabilities_excited,
                                    artificial_detuning,
                                    damped_osc_freq_lim: int):
            """
            Fits damped_osc_func on the data 50 times with random initial parameters,
            and returns the lmfit result with the lowest BIC. The fitted frequency
            is bounded in [artificial_detuning/4, damped_osc_freq_lim*artificial_detuning].
            """
            
            damped_oscillation_fit = None
            best_bic = np.inf
            for fit_trial_idx in range(50):
                damped_osc_model = Model(damped_osc_func)
                damped_osc_params = damped_osc_model.make_params(
                    tau = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency= np.random.uniform(0.5, damped_osc_freq_lim/4) * artificial_detuning,
                    phase=np.random.uniform(0.5, 1) * np.pi,
                    amplitude=np.random.uniform(0.4, 0.6),
                    offset=np.random.uniform(0.4, 0.6)
                )
                damped_osc_params['tau'].vary, damped_osc_params['tau'].min, damped_osc_params['tau'].max = (
                    True, 0, 5*measurement_times[-1])
                damped_osc_params['frequency'].vary, damped_osc_params['frequency'].min, damped_osc_params['frequency'].max = (
                    True, artificial_detuning/4, damped_osc_freq_lim*artificial_detuning)
                damped_osc_params['phase'].vary, damped_osc_params['phase'].min, damped_osc_params['phase'].max = (
                    True, -np.pi, np.pi)
                damped_osc_params['amplitude'].vary, damped_osc_params['amplitude'].min, damped_osc_params['amplitude'].max = (
                    True, 0.4, 0.6)
                damped_osc_params['offset'].vary, damped_osc_params['offset'].min, damped_osc_params['offset'].max = (
                    True, 0.1, 0.9)
                damped_oscillation_trial_fit = damped_osc_model.fit(
                    probabilities_excited,
                    damped_osc_params,
                    t=measurement_times
                )
                if damped_oscillation_trial_fit.bic < best_bic:
                    best_bic = damped_oscillation_trial_fit.bic
                    damped_oscillation_fit = damped_oscillation_trial_fit
            return damped_oscillation_fit
        
        def fit_double_dumped_osc_function(measurement_times,
                                           probabilities_excited,
                                           artificial_detuning):
            """
            Fits double_damped_osc_func on the data 30 times with random initial
            parameters, and returns the lmfit result with the lowest BIC.
            """

            double_damped_oscillation_fit = None
            best_bic = np.inf
            for fit_trial_idx in range(30):
                double_damped_osc_model = Model(double_damped_osc_func)
                double_damped_osc_params = double_damped_osc_model.make_params(
                    tau_1 = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency_1 = np.random.uniform(0.5, 2.5)*artificial_detuning,
                    phase_1=0.0,
                    amplitude_1=0.5*(max(probabilities_excited) - min(probabilities_excited)),

                    tau_2 = np.random.uniform(0.2, 2) * measurement_times[-1],
                    frequency_2 = np.random.uniform(0.5, 2.5)*artificial_detuning,
                    phase_2=0.0,
                    amplitude_2=0.5*(max(probabilities_excited) - min(probabilities_excited)),

                    offset=0.5
                )
                double_damped_osc_params['tau_1'].vary, double_damped_osc_params['tau_1'].min, double_damped_osc_params['tau_1'].max = (
                    True, 0.05*measurement_times[-1], 5*measurement_times[-1])
                double_damped_osc_params['frequency_1'].vary, double_damped_osc_params['frequency_1'].min, double_damped_osc_params['frequency_1'].max = (
                    True, artificial_detuning/4, 4*artificial_detuning)
                double_damped_osc_params['phase_1'].vary, double_damped_osc_params['phase_1'].min, double_damped_osc_params['phase_1'].max = (
                    True, -np.pi, np.pi)
                double_damped_osc_params['amplitude_1'].vary, double_damped_osc_params['amplitude_1'].min, double_damped_osc_params['amplitude_1'].max = (
                    True, 0.1, 0.9)
                
                double_damped_osc_params['tau_2'].vary, double_damped_osc_params['tau_2'].min, double_damped_osc_params['tau_2'].max = (
                    True, 0.05*measurement_times[-1], 5*measurement_times[-1])
                double_damped_osc_params['frequency_2'].vary, double_damped_osc_params['frequency_2'].min, double_damped_osc_params['frequency_2'].max = (
                    True, artificial_detuning/4, 4*artificial_detuning)
                double_damped_osc_params['phase_2'].vary, double_damped_osc_params['phase_2'].min, double_damped_osc_params['phase_2'].max = (
                    True, -np.pi, np.pi)
                double_damped_osc_params['amplitude_2'].vary, double_damped_osc_params['amplitude_2'].min, double_damped_osc_params['amplitude_2'].max = (
                    True, 0.1, 0.9)
                double_damped_osc_params['offset'].vary, double_damped_osc_params['offset'].min, double_damped_osc_params['offset'].max = (
                    True, 0.48, 0.52)
                double_damped_oscillation_trial_fit = double_damped_osc_model.fit(
                    probabilities_excited,
                    double_damped_osc_params,
                    t=measurement_times
                )
                if double_damped_oscillation_trial_fit.bic < best_bic:
                    best_bic = double_damped_oscillation_trial_fit.bic
                    double_damped_oscillation_fit = double_damped_oscillation_trial_fit
            return double_damped_oscillation_fit
        
        self.T2_echo_values = {}
        bic_diff_tolerance = 0.30
        result_probs_per_n_qubits = self.result_probs_per_n_qubits
        probabilities_excited_per_n_qubits = []
        artificial_detunings = {'original': self.artificial_detunings.copy(),
                                'minus': self.artificial_detunings.copy(),
                                'plus': self.artificial_detunings.copy()}
        damped_osc_curves = {'original': [], 'minus': [], 'plus': []}
        double_damped_osc_curves = {'original': [], 'minus': [], 'plus': []}
        damped_osc_fits = {'minus': [], 'plus': []}
        double_damped_osc_fits = {'minus': [], 'plus': []}
        bic_avgs = {'minus': [], 'plus': []}
        bic_diffs = {'minus': [], 'plus': []}
        tau_fits = {'original': [], 'minus': [], 'plus': []}
        tau_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        tau_1_fits = {'original': [], 'minus': [], 'plus': []}
        tau_1_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        tau_2_fits = {'original': [], 'minus': [], 'plus': []}
        tau_2_fits_stderr = {'original': [], 'minus': [], 'plus': []}
        frequency_1_fits = {'original': [], 'minus': [], 'plus': []}
        frequency_2_fits = {'original': [], 'minus': [], 'plus': []}
        fit_bics = {'single_beating': [], 'double_beating': [], 'single_minus': [], 'single_plus': []}
        xf_data_per_n_qubits = []
        fft_data_amplitude_per_n_qubits = []
        xf_fit_per_n_qubits = []
        fft_fit_amplitude_per_n_qubits = []
        peak_freqs_per_n_qubits = []
        peak_amps_per_n_qubits = []
        fitted_double_osc_pass = [False for qubit_idx in range(len(self.qubit_list))]

        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [result_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                     for entry in range(len(measurement_times))]
            probabilities_excited_per_n_qubits.append(probabilities_excited)

            # Iterate 30 times for best fit
            damped_oscillation_fit = fit_dumped_osc_function(measurement_times,
                                                             probabilities_excited,
                                                             artificial_detunings['original'][qubit_idx],
                                                             damped_osc_freq_lim=13)
            tau_fit, frequency_fit, phase_fit, amplitude_fit, offset_fit = (damped_oscillation_fit.params['tau'].value,
                damped_oscillation_fit.params['frequency'].value, damped_oscillation_fit.params['phase'].value,
                damped_oscillation_fit.params['amplitude'].value, damped_oscillation_fit.params['offset'].value)
            tau_stderr = damped_oscillation_fit.params['tau'].stderr
            measurement_times_fit = np.linspace(measurement_times[0],
                                                measurement_times[-1],
                                                num = 2001)
            damped_oscillation_curve = damped_osc_func(measurement_times_fit, tau_fit, frequency_fit,
                                                       phase_fit, amplitude_fit, offset_fit)
            tau_fits['original'].append(tau_fit)
            tau_fits_stderr['original'].append(tau_stderr)
            damped_osc_curves['original'].append(damped_oscillation_curve)
            self.T2_echo_values[f'{self.qubit_labels[qubit_idx]} [us]'] = np.round(1e6 * tau_fit, 1)

            # Iterate 30 times for best fit
            double_damped_oscillation_fit = fit_double_dumped_osc_function(measurement_times,
                                                                           probabilities_excited,
                                                                           artificial_detunings['original'][qubit_idx])
            (tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
             tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
             offset_fit) = (
                            double_damped_oscillation_fit.params['tau_1'].value, double_damped_oscillation_fit.params['frequency_1'].value, 
                            double_damped_oscillation_fit.params['phase_1'].value, double_damped_oscillation_fit.params['amplitude_1'].value,
                            double_damped_oscillation_fit.params['tau_2'].value, double_damped_oscillation_fit.params['frequency_2'].value, 
                            double_damped_oscillation_fit.params['phase_2'].value, double_damped_oscillation_fit.params['amplitude_2'].value,
                            damped_oscillation_fit.params['offset'].value)
            (tau_1_stderr, tau_2_stderr) = (double_damped_oscillation_fit.params['tau_1'].stderr,
                                            double_damped_oscillation_fit.params['tau_2'].stderr)
            double_damped_oscillation_curve = double_damped_osc_func(measurement_times_fit,
                                                                     tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
                                                                     tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
                                                                     offset_fit)
            tau_1_fits['original'].append(tau_1_fit)
            tau_2_fits['original'].append(tau_2_fit)
            tau_1_fits_stderr['original'].append(tau_1_stderr)
            tau_2_fits_stderr['original'].append(tau_2_stderr)
            frequency_1_fits['original'].append(frequency_1_fit)
            frequency_2_fits['original'].append(frequency_2_fit)
            double_damped_osc_curves['original'].append(double_damped_oscillation_curve)

            pad_N = 5000
            N_data = len(measurement_times)
            T_data = measurement_times[1] - measurement_times[0]
            signal_data = probabilities_excited - np.mean(probabilities_excited) # remove DC component for FFT
            window_data = np.hanning(N_data)
            signal_data = signal_data * window_data # apply window to data
            yf_data = fft(signal_data, n=pad_N)
            xf_data = fftfreq(pad_N, T_data)[:pad_N//2]
            fft_data_amplitude = 2.0 * np.abs(yf_data[:pad_N//2]) / np.sum(window_data)
            xf_data_per_n_qubits.append(xf_data)
            fft_data_amplitude_per_n_qubits.append(fft_data_amplitude)

            signal_fit = double_damped_osc_func(measurement_times,
                                                tau_1_fit, frequency_1_fit, phase_1_fit, amplitude_1_fit,
                                                tau_2_fit, frequency_2_fit, phase_2_fit, amplitude_2_fit,
                                                offset_fit)
            signal_fit -= np.mean(signal_fit)
            window_fit = np.hanning(N_data)
            signal_fit = signal_fit * window_fit # apply window to data
            yf_fit = fft(signal_fit, n=pad_N)
            xf_fit = fftfreq(pad_N, T_data)[:pad_N//2]
            fft_fit_amplitude = 2.0 * np.abs(yf_fit[:pad_N//2]) / np.sum(window_fit)
            xf_fit_per_n_qubits.append(xf_fit)
            fft_fit_amplitude_per_n_qubits.append(fft_fit_amplitude)


            peaks, properties = find_peaks(fft_data_amplitude,
                                           prominence=0.1 * np.max(fft_data_amplitude))
            peak_freqs = xf_data[peaks]
            peak_amps = fft_data_amplitude[peaks]
            peak_freqs_per_n_qubits.append(peak_freqs)
            peak_amps_per_n_qubits.append(peak_amps)

            fit_bics['single_beating'].append(damped_oscillation_fit.bic)
            fit_bics['double_beating'].append(double_damped_oscillation_fit.bic)
            bic_avg = (damped_oscillation_fit.bic+double_damped_oscillation_fit.bic) /2
            bic_diff = np.abs(damped_oscillation_fit.bic-double_damped_oscillation_fit.bic) / np.abs(bic_avg)
            if bic_diff > bic_diff_tolerance or len(peaks)>1: # if bic difference is larger than bic_diff_tolerance
                                                # or more than 1 peaks have been detected in the FFT
                fitted_double_osc_pass[qubit_idx] = True
                original_detuning = artificial_detunings['original'][qubit_idx].copy()
                f1 = frequency_1_fit
                f2 = frequency_2_fit
                artificial_detunings['minus'][qubit_idx] = np.abs(original_detuning - (f1+f2)/2)
                artificial_detunings['plus'][qubit_idx] = original_detuning + (f1+f2)/2

        if any(fitted_double_osc_pass):
            double_beating_labels = []
            for qubit_idx in range(len(self.qubit_list)):
                if fitted_double_osc_pass[qubit_idx] == True:
                    double_beating_labels.append(self.qubit_labels[qubit_idx])
            timestamp_now = datetime.now().replace(microsecond=0)
            print(f"[{timestamp_now}] "
                "Detected double beating for qubits "
                f"{double_beating_labels}.")
            qc_minus = self._quantum_circuit(backend=self.backend,
                                        qubit_list=self.qubit_list,
                                        measurement_times=measurement_times,
                                        artificial_detunings=artificial_detunings['minus'])
            qc_plus = self._quantum_circuit(backend=self.backend,
                                        qubit_list=self.qubit_list,
                                        measurement_times=measurement_times,
                                        artificial_detunings=artificial_detunings['plus'])
            print(f"[{timestamp_now}] "
                "Re-launching the measurement routine with adjusted artificial detunings to get rid of the double beating(s) ... ")
            super().__init__(backend=self.backend,
                                qubit_list=self.qubit_list,
                                qc=[qc_minus, qc_plus],
                                num_shots=self.num_shots,
                                store_record=True)   
                               
        for qubit_idx in range(len(self.qubit_list)):    
            if fitted_double_osc_pass[qubit_idx] == True:
                config = ['minus', 'plus']
                multi_probabilities_excited = {}
                for idx in range(2):

                    probabilities_excited_new = [self.result_multi_probs_per_n_qubits[idx][qubit_idx][entry]['1'] \
                                                for entry in range(len(measurement_times))]
                    multi_probabilities_excited[config[idx]] = probabilities_excited_new
                    damped_osc_fits[config[idx]] = fit_dumped_osc_function(measurement_times,
                                                                          probabilities_excited_new,
                                                                          artificial_detunings[config[idx]][qubit_idx],
                                                                          damped_osc_freq_lim=20)
                    double_damped_osc_fits[config[idx]] = fit_double_dumped_osc_function(measurement_times,
                                                                                         probabilities_excited_new,
                                                                                         artificial_detunings[config[idx]][qubit_idx])
                    bic_avgs[config[idx]] = (damped_osc_fits[config[idx]].bic+double_damped_osc_fits[config[idx]].bic) /2
                    bic_diffs[config[idx]] = np.abs(damped_osc_fits[config[idx]].bic-double_damped_osc_fits[config[idx]].bic) \
                                            / np.abs(bic_avgs[config[idx]])

                if bic_diffs['minus'] < bic_diffs['plus']:
                    winning_config = 'minus'
                else: winning_config = 'plus'
                tau_fits[winning_config], frequency_fits, phase_fit_new, amplitude_fit_new, offset_fit_new = (
                damped_osc_fits[winning_config].params['tau'].value, damped_osc_fits[winning_config].params['frequency'].value,
                damped_osc_fits[winning_config].params['phase'].value, damped_osc_fits[winning_config].params['amplitude'].value,
                damped_osc_fits[winning_config].params['offset'].value)
                tau_stderr = damped_osc_fits[winning_config].params['tau'].stderr
                tau_fits_stderr[winning_config].append(tau_stderr)
                damped_osc_curve = damped_osc_func(measurement_times_fit, tau_fits[winning_config], frequency_fits,
                                                    phase_fit_new, amplitude_fit_new, offset_fit_new)
                self.T2_echo_values[f'{self.qubit_labels[qubit_idx]} [us]'] = np.round(1e6 * tau_fits[winning_config], 1)
                print(tau_fits[winning_config], frequency_fits, phase_fit_new, amplitude_fit_new, offset_fit_new)

                fig, axes = plt.subplot_mosaic(
                            [
                                ["A", "B"],
                                ["C", "C"]
                            ],
                            figsize=(12, 12),
                            dpi=300
                        )
                ax = axes["A"]
                ax1 = axes["B"]
                ax2 = axes["C"]
                fig.suptitle(
                    f'T2 Echo measurement'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.record.date_timestamp}_{self.record.job_timestamp}',
                    fontsize=14
                )
            else:
                fig, ax = plt.subplots(dpi=300)
            ax.plot(1e6*measurement_times,
                    probabilities_excited_per_n_qubits[qubit_idx],
                    label=f'{self.qubit_labels[qubit_idx]}: T2 echo = {1e6 * tau_fits['original'][qubit_idx]:.1f} μs',
                    alpha=0.6,
                    color = f'C{qubit_idx}',
                    marker='o')
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel(r'Population $|1\rangle$')
            ax.set_ylim(-0.05, 1.05)
            ax.plot(1e6*measurement_times_fit,
                        damped_osc_curves['original'][qubit_idx],
                        alpha=0.7,
                        color = 'red',
                        label = 'Single oscillation fit')
            if fitted_double_osc_pass[qubit_idx] == True:
                ax.plot(1e6*measurement_times_fit,
                            double_damped_osc_curves['original'][qubit_idx],
                            alpha=0.7,
                            color = 'green',
                            label = 'Double oscillation fit')
                ax.plot([], [], ' ', label=' ')
                ax.plot([], [], ' ', label=f'Artificial detuning ' + r'$f = $' + f'{artificial_detunings['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$f_1 = $' + f'{frequency_1_fits['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$f_2 = $' + f'{frequency_2_fits['original'][qubit_idx]:.1f} Hz')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$T_2^{\text{echo}(1)} = $' + f'{1e6 * tau_1_fits['original'][qubit_idx]:.1f} μs')
                ax.plot([], [], ' ', label=f'Double osc. fit ' + r'$T_2^{\text{echo}(2)} = $' + f'{1e6 * tau_2_fits['original'][qubit_idx]:.1f} μs')
                ax.set_title("Initial measurement")
                ax.legend(loc='center left', bbox_to_anchor=(-0.9, 0.75))

                ax1.plot(xf_data_per_n_qubits[qubit_idx],
                         fft_data_amplitude_per_n_qubits[qubit_idx],
                        label=f'{self.qubit_labels[qubit_idx]} data',
                        alpha=0.8,
                        color=f'C{qubit_idx}',
                        marker='o',
                        markersize=0.5)
                ax1.plot(xf_fit_per_n_qubits[qubit_idx],
                         fft_fit_amplitude_per_n_qubits[qubit_idx],
                        label=f'{self.qubit_labels[qubit_idx]} double osc. fit',
                        alpha=0.6,
                        color='green')
                ax1.axvline(frequency_1_fits['original'][qubit_idx],
                        label='Double osc. fit ' + r'$f_1$',
                        linestyle = '--',
                        linewidth = 1.2,
                        alpha = 0.7,
                        color='black')
                ax1.axvline(frequency_2_fits['original'][qubit_idx],
                        label='Double osc. fit ' + r'$f_2$',
                        linestyle='--',
                        linewidth = 1.2,
                        alpha = 0.9,
                        color='orange')
                ax1.scatter(
                            peak_freqs_per_n_qubits[qubit_idx],
                            peak_amps_per_n_qubits[qubit_idx],
                            marker='*',
                            s=150,
                            color='orange',
                            edgecolors='black',
                            linewidths=1.5,
                            label=f'Detected peaks (on {self.qubit_labels[qubit_idx]} data)',
                            zorder=10
                        )
                ax1.set_title("FFT of initial measurement data")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Amplitude")
                ax1.set_xlim(0, xf_data[-1]) # restrict frequency values to those of the data
                ax1.legend(loc='center right', bbox_to_anchor=(1.65, 0.85))

                markers = {'minus': 's', 'plus': '^'}
                colors = {'minus': 'darkred', 'plus': 'purple'}
                sign = {'minus': '-', 'plus': '+'}
                ax2.plot(1e6*measurement_times,
                        multi_probabilities_excited[winning_config],
                        label=f'{self.qubit_labels[qubit_idx]}: T2 echo = {1e6 * tau_fits[winning_config]:.1f} μs',
                        alpha=0.6,
                        color = f'C{qubit_idx}',
                        marker=markers[winning_config])
                ax2.plot(1e6*measurement_times_fit,
                        damped_osc_curve,
                        alpha=0.7,
                        color = colors[winning_config],
                        label = 'Single oscillation fit')
                ax2.plot([], [], ' ', label=' ')
                ax2.plot([], [], ' ', label=f'Artificial detuning: ' + r"$f' = $" + f'{artificial_detunings[winning_config][qubit_idx]:.1f} Hz')
                ax2.plot([], [], ' ', label=' ')
                ax2.plot([], [], ' ', label=f'Artificial detuning adjustment:')
                ax2.plot([], [], ' ', label=r"$f' = f$"+f'{sign[winning_config]}'+r'$(f_1+f_2)/2$')
                ax2.set_title('Final measurement with adjusted artificial detuning')
                ax2.set_xlabel('Time (μs)')
                ax2.set_ylabel(r'Population $|1\rangle$')
                ax2.set_ylim(-0.05, 1.05)
                ax2.legend(loc='center left', bbox_to_anchor=(-0.43, 0.8))
            else:
                ax.set_title(
                f'T2 Echo measurement'
                f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                f'\n{self.record.date_timestamp}_{self.record.job_timestamp}')
                ax.legend()

            T2_echo_fig_path = (
                Path(self.record.project_dir)
                / f"T2_Echo_plot_{self.qubit_labels[qubit_idx]}_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(T2_echo_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["Measurement times [s]"] = measurement_times
        self.experiment_data["Artificial detunings [Hz]"] = artificial_detunings['original']
        self.experiment_data["Artificial detunings minus [Hz]"] = artificial_detunings['minus']
        self.experiment_data["Artificial detunings plus [Hz]"] = artificial_detunings['plus']
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["T2 echo values"] = self.T2_echo_values
        json_file_path = (
            Path(self.record.project_dir)
            / f"T2_echo_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

class FlippingMeasurement(BaseMeasurement):
    """
    This measurement class implements a flipping experiment, used for the fine
    calibration of the amplitude of the pi (or pi/2) pulses. With equator = True,
    the qubit is first brought to the equator of the Bloch sphere with a pi/2 pulse,
    followed by an even number N of (effective) pi pulses, prior to measurement.

    Ideally, the qubit remains on the equator for any N, with prob(|1>) = 0.5.
    A small amplitude error accumulates coherently with every additional pulse,
    amplifying the error syndrome, while the qubit being on the equator makes the
    measurement first-order sensitive to it [1]. With equator = False, the
    sequence is first-order insensitive to amplitude errors [2].

    The data are fitted with both a line and a cosine, with the best fit chosen
    based on the Bayesian Information Criterion (BIC). From each fit a scale factor
    is extracted, by which the pulse amplitude should be multiplied in order to
    correct for the amplitude error [2].

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 5, section 5.2.4.
    [2] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Transmon.measure_flipping() and
        analysis_v2/timedomain_analysis.py, FlippingAnalysis.get_scale_factor()
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 max_number_of_flips: int = 30,
                 equator: bool = True,
                 rotation_axis: str = 'x', # 'x' or 'y'
                 rotation_angle: str = '180', # '180' or '90'
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            num_shots (int):
                The number of shots for which the experiment is executed.

            max_number_of_flips (int):
                The maximum number of (effective) pi pulses, which must be an
                even integer greater than 0. The number of flips is swept as
                [0, 2, 4, ..., max_number_of_flips].

            equator (bool):
                If True, an initial pi/2 pulse brings the qubit to the equator
                of the Bloch sphere.

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.

            rotation_angle (str):
                Either '180' for pi pulses, or '90' for pairs of pi/2 pulses,
                each pair forming an effective pi pulse.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        if max_number_of_flips % 2 != 0 or max_number_of_flips <= 0:
            raise ValueError("max_number_of_flips must be an even integer greater than 0!")
        assert rotation_angle in ['90', '180']
        assert rotation_axis.lower() in ['x', 'y']

        number_of_flips = np.arange(0, max_number_of_flips+1, 2)
        self.equator = equator
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        
        qc = self._quantum_circuit(backend,
                                   qubit_list,
                                   number_of_flips,
                                   equator,
                                   rotation_axis,
                                   rotation_angle)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points=use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(number_of_flips)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        
    def _quantum_circuit(self,
                         backend,
                         qubit_list: list,
                         number_of_flips: np.array,
                         equator: bool = True,
                         rotation_axis: str = 'x', # 'x' or 'y'
                         rotation_angle: str = '180'): # '180' or '90'
        """
        This instance method creates the experiment quantum circuit. For each
        number of flips N, a measurement block resets the qubits, optionally
        applies a pi/2 pulse (equator = True), applies N (effective) pi pulses
        and measures all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            number_of_flips (np.array):
                The number of (effective) pi pulses of each measurement block.

            equator (bool):
                If True, an initial pi/2 pulse brings the qubit to the equator
                of the Bloch sphere.

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.

            rotation_angle (str):
                Either '180' for pi pulses, or '90' for pairs of pi/2 pulses,
                each pair forming an effective pi pulse.
        """

        bit_idx = 0
        self.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in qubit_list]
        qc = QuantumCircuit(backend.num_qubits,
                            len(number_of_flips)*len(qubit_list),
                            name=f'Flipping_{backend.name}')

        for i, n in enumerate(number_of_flips):
            for qubit_idx in qubit_list:
                qc.reset(qubit_idx)
            qc.barrier()

            if equator == True:
                for qubit_idx in qubit_list:
                    if rotation_axis == 'y':
                        qc.ry(np.pi/2, qubit_idx)
                    else:
                        qc.rx(np.pi/2, qubit_idx)
                qc.barrier()

            for dummy_idx in range(n):
                for qubit_idx in qubit_list:
                    if rotation_angle == '90':
                        if rotation_axis == 'y':
                            qc.ry(np.pi/2, qubit_idx)
                            qc.ry(np.pi/2, qubit_idx)
                        else:
                            qc.rx(np.pi/2, qubit_idx)
                            qc.rx(np.pi/2, qubit_idx)
                    else:
                        if rotation_axis == 'y':
                            qc.ry(np.pi, qubit_idx)
                        else:
                            qc.rx(np.pi, qubit_idx)
                qc.barrier()

            for qubit_idx in qubit_list:
                qc.measure(qubit = qubit_idx, cbit = bit_idx)
                bit_idx += 1
            qc.barrier()
            
        return qc
    
    def _data_analysis(self,
                       number_of_flips):
        """
        This instance method fits the excited state population of each qubit with
        both a line f(x) = a*x + b and a cosine f(x) = a*cos(bx + c) + 0.5 (see
        fitting_functions.py), and computes the corresponding amplitude scale
        factors, 1 / (1 - 4*f_line) with f_line = -a/(2*pi) for the line fit, and
        1 / (1 + 2*f_cos) with f_cos = sign * b/(2*pi) for the cosine fit, where the
        sign of f_cos is resolved from the fitted phase c. The scale factors, the BIC of each fit and
        the best fit are stored in the attribute 'flipping_parameters', while the
        plots and the experiment data are stored in the project directory.

        Args:
            number_of_flips (np.array):
                The number of (effective) pi pulses of each measurement block.
        """

        cos_fit_axis = np.linspace(0, number_of_flips[-1], 300)
        self.flipping_parameters = {f'Q{qubit_idx}': {} for qubit_idx in self.qubit_list}

        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.result_probs_per_n_qubits[qubit_idx][entry]['1'] \
                                    for entry in range(len(number_of_flips))]

            linear_model = Model(linear_func)
            linear_params = linear_model.make_params(
                a=0.0,
                b=0.5
            )
            linear_params['b'].vary = True
            linear_fit = linear_model.fit(
                probabilities_excited,
                linear_params,
                x=number_of_flips
            )
            a_linear_fit, b_linear_fit = linear_fit.params['a'].value, linear_fit.params['b'].value
            linear_curve = linear_func(number_of_flips, a_linear_fit, b_linear_fit)
            linear_freq = - a_linear_fit / (2*np.pi)
            scale_factor_line = 1 / (1 - 4 * linear_freq)
            self.flipping_parameters[self.qubit_labels[qubit_idx]]['Line scale factor [a.u.]'] = scale_factor_line

            cos_model = Model(cos_func)
            cos_params = cos_model.make_params(
                amplitude=0.45,
                frequency=0.1,
                phase=0.0,
                offset=0.5 # for d fixed to 0.5, a can only be in [0.0, 0.5]
            )
            cos_params['amplitude'].vary = True
            cos_params['amplitude'].min=0.05
            cos_params['amplitude'].max=0.5
            cos_params['frequency'].vary = True
            cos_params['frequency'].min = 0.05
            cos_params['phase'].vary = True
            cos_params['offset'].vary = False
            cos_fit = cos_model.fit(
                probabilities_excited,
                cos_params,
                x=number_of_flips
            )
            a_cos_fit, b_cos_fit, c_cos_fit, d_cos_fit = (cos_fit.params['amplitude'].value, cos_fit.params['frequency'].value,
                                                        cos_fit.params['phase'].value, cos_fit.params['offset'].value)
            cosine_curve = cos_func(cos_fit_axis, a_cos_fit, b_cos_fit, c_cos_fit, d_cos_fit)
            cos_frequency = (b_cos_fit / (2*np.pi)) * np.sign(np.mod(c_cos_fit, 2*np.pi) - np.pi)
            scale_factor_cos = 1 / (1 + 2*cos_frequency)
            self.flipping_parameters[self.qubit_labels[qubit_idx]]['Cos scale factor [a.u.]'] = scale_factor_cos

            if linear_fit.bic < cos_fit.bic:
                best_fit = 'Line'
            else:
                best_fit = 'Cos'
            self.flipping_parameters[self.qubit_labels[qubit_idx]]['Linear fit BIC'] = linear_fit.bic
            self.flipping_parameters[self.qubit_labels[qubit_idx]]['Cos fit BIC'] = cos_fit.bic
            self.flipping_parameters[self.qubit_labels[qubit_idx]]['Best fit'] = best_fit

            fig, ax = plt.subplots(dpi=300)
            for ax_obj in [ax_all, ax]:
                ax_obj.plot(number_of_flips,
                        probabilities_excited,
                        alpha=0.6,
                        color = f'C{qubit_idx}',
                        marker='o',
                        label = f'Qubit {self.qubit_labels[qubit_idx]}')
                
                ax_obj.set_xlabel(r'Number of (effective) $\pi$ pulses')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(
                    f'Flipping measurement'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.date_timestamp}_{self.job_timestamp}')
                ax_obj.set_ylim(-0.05, 1.05)
                
            ax.plot(number_of_flips,
                    linear_curve,
                    alpha=0.8,
                    color = 'red',
                    label = 'Line fit',
                    linestyle='--',
                    linewidth=1.2)
            ax.plot(cos_fit_axis,
                    cosine_curve,
                    alpha=0.8,
                    color = 'purple',
                    label = 'Cos fit',
                    linestyle='--',
                    linewidth=1.2)

            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=f'Best fit: {best_fit}')
            ax.plot([], [], ' ', label=f'Line scale factor: {scale_factor_line:.4f}')
            ax.plot([], [], ' ', label=f'Cos scale factor: {scale_factor_cos:.4f}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.73))
            flipping_fig_path = (
                Path(self.project_dir)
                / f"flipping_plot_{self.qubit_labels[qubit_idx]}_{self.date_timestamp}_{self.job_timestamp}.png"
            )
            fig.savefig(flipping_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        flipping_all_fig_path = (
            Path(self.project_dir)
            / f"flipping_plot_ALL_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig_all.savefig(flipping_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.date_timestamp}_{self.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["Number of flips"] = number_of_flips
        self.experiment_data["Equator"] = self.equator
        self.experiment_data["Rotation axis"] = self.rotation_axis
        self.experiment_data["Rotation angle"] = self.rotation_angle
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["Flipping parameters"] = self.flipping_parameters
        json_file_path = (
            Path(self.project_dir)
            / f"flipping_data_{self.date_timestamp}_{self.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        number_of_flips = json_data['Number of flips']
        cls.equator = json_data['Equator']
        cls.rotation_axis = json_data['Rotation axis']
        cls.rotation_angle = json_data['Rotation angle']
        cls.qubit_list = json_data['Qubit list']
        cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls,
                           number_of_flips)
        return json_data
     

class AllXYMeasurement(BaseMeasurement):
    """
    This measurement class implements the AllXY experiment, a sensitive test of
    the single-qubit gates. All 21 combinations of two consecutive rotations,
    taken from the identity and the x- and y-axis rotations by an angle of pi/2
    or pi, are applied to the qubit prior to measurement, each combination
    measured twice (42 measurement blocks in total) [1, 3].

    Ideally, the qubit ends up either in the ground state (first 5 combinations),
    on the equator of the Bloch sphere (next 12 combinations) or in the excited
    state (last 4 combinations), giving a characteristic "staircase" pattern for
    prob(|1>) of 0, 0.5 and 1. Different gate errors, such as amplitude errors and
    detuning, produce distinct deviations from the ideal staircase, providing a
    quick diagnosis of the single-qubit gates [1].

    As a figure of merit, the mean absolute deviation from the ideal staircase
    is extracted [2].

    In the syndrome labels, 'X' ('Y') denotes a pi rotation and 'x' ('y') a pi/2
    rotation around the x (y) axis, while 'I' denotes the identity.

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 5, section 5.2.3.
    [2] Control for Programmable Superconducting Quantum Systems,
        Michiel Adriaan Rol, PhD Thesis, Delft University of Technology (2020)
        https://doi.org/10.4233/uuid:0a2ba212-f6bf-4c64-8f3d-b707f1e44953
        Chapter 5, section 5.2.3.
    [3] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Transmon.measure_allxy()
    """

    def __init__(self,
                 backend,
                 qubit_list: list,
                 num_shots: int,
                 use_ro_cal_points: bool = True,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].

            num_shots (int):
                The number of shots for which the experiment is executed.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        qc = self._quantum_circuit(backend,
                                   qubit_list)
        
        super().__init__(backend=backend,
                         qubit_list=qubit_list,
                         qc=qc,
                         num_shots=num_shots,
                         use_ro_cal_points=use_ro_cal_points,
                         directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis()
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        
    def _quantum_circuit(self,
                         backend,
                         qubit_list):
        """
        This instance method creates the experiment quantum circuit. For each of
        the 21 pulse combinations (repeated twice), a measurement block resets
        the qubits, applies the two rotations and measures all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_list (list):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
        """

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
                            name=f'AllXY_{backend.name}')

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
        """
        This instance method compares the excited state population of each qubit
        with the ideal AllXY staircase, and stores the mean absolute deviation in
        the attribute 'allxy_deviations' (ideal value 0). The plots and the
        experiment data are stored in the project directory.
        """

        AllXY_syndrome_labels = ['II', 'XX', 'YY', 'XY', 'YX', 'xI', 'yI', 'xy',
                                 'yx', 'xY', 'yX', 'Xy', 'Yx', 'xX', 'Xx', 'yY',
                                 'Yy', 'XI', 'YI', 'xx', 'yy']
        self.allxy_deviations = {}

        fig_all, ax_all = plt.subplots(dpi=300)
        for qubit_idx in range(len(self.qubit_list)):
            probabilities_excited = [self.result_probs_per_n_qubits[qubit_idx][entry]['1'] \
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
                        marker='o')
                ax_obj.set_xlabel('Syndrome')
                ax_obj.set_ylabel(r'Population $|1\rangle$')
                ax_obj.set_title(
                    f'AllXY measurement'
                    f'\n{self.backend_name} {self.backend_type} | num_shots = {self.num_shots}'
                    f'\n{self.date_timestamp}_{self.job_timestamp}')
        
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
                    alpha=0.7,
                    color = 'red')
            ax.legend()
            allxy_fig_path = (
                Path(self.project_dir)
                / f"allxy_plot_{self.qubit_labels[qubit_idx]}_{self.date_timestamp}_{self.job_timestamp}.png"
            )
            fig.savefig(allxy_fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        ax_all.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        allxy_all_fig_path = (
            Path(self.project_dir)
            / f"allxy_plot_ALL_{self.date_timestamp}_{self.job_timestamp}.png"
        )
        fig_all.savefig(allxy_all_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_all)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.date_timestamp}_{self.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit list"] = self.qubit_list
        self.experiment_data["AllXY syndromes"] = AllXY_syndrome_labels
        self.experiment_data["Processed data"] = {f"{self.qubit_labels[qubit_idx]}":self.result_probs_per_n_qubits[qubit_idx] \
                                                 for qubit_idx in range(len(self.qubit_list))}
        self.experiment_data["AllXY deviations"] = self.allxy_deviations
        json_file_path = (
            Path(self.project_dir)
            / f"allxy_data_{self.date_timestamp}_{self.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        cls.qubit_list = json_data['Qubit list']
        cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls)
        return json_data

class ConditionalOscMeasurement(BaseMeasurement):
    """
    This measurement class implements the conditional oscillation experiment,
    used for characterizing the two-qubit CZ gate by measuring its conditional
    phase and estimating its leakage [1, 2].

    For each qubit pair [q_targ, q_ctrl], two variants of a Ramsey-like experiment
    are performed. In the first variant (Off), q_targ is brought to the equator of
    the Bloch sphere with a pi/2 pulse while q_ctrl remains in the ground state,
    after which the CZ gate is applied. A recovery pi/2 pulse, around an axis in
    the equatorial plane set by a swept phase phi, is then applied on q_targ before
    measuring both qubits simultaneously. In the second variant (On), q_ctrl is
    additionally excited with a pi pulse before the CZ gate, and returned to the
    ground state with a second pi pulse before the measurement [1].

    The population of q_targ oscillates with phi in both variants, and the
    conditional phase corresponds to the phase difference between the two
    oscillations, ideally 180 degrees for a single CZ gate [1]. Multiple CZ
    repetitions can be used to amplify phase errors. The missing fraction m is
    the difference in the average excited state population of q_ctrl between the
    On and Off variants, from which the leakage of the CZ gate can be estimated as
    L1 ~ m/2 [1, 3].

    Further reading:
    [1] Control for Programmable Superconducting Quantum Systems,
        Michiel Adriaan Rol, PhD Thesis, Delft University of Technology (2020)
        https://doi.org/10.4233/uuid:0a2ba212-f6bf-4c64-8f3d-b707f1e44953
        Chapter 7, section 7.5.4, and Chapter 3, section 3.4.1.
    [2] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 7, section 7.2.
    [3] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Device.measure_conditional_oscillation() and
        analysis_v2/timedomain_analysis.py, Conditional_Oscillation_Analysis
    """

    def __init__(self,
                backend,
                qubit_pairs: list[list[int]],
                num_shots: int,
                num_angles: int = 19,
                cz_repetitions: int = 1,
                use_ro_cal_points: bool = True,
                directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_pairs (list(list)):
                An ordered list of qubit pairs [q_targ, q_ctrl] to be measured
                simultaneously, where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            num_shots (int):
                The number of shots for which the experiment is executed.

            num_angles (int):
                The number of phases phi of the recovery pi/2 pulse, equally
                spaced in [0, 2*pi].

            cz_repetitions (int):
                The number of consecutive CZ gates applied.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        qubit_list = []
        for qubit_pair in qubit_pairs:
            for qubit_idx in qubit_pair:
                if qubit_idx in qubit_list:
                    raise ValueError("A single qubit cannot be used in multiple pairs at the same time!")
                else:
                    qubit_list.append(qubit_idx)
        
        angles = np.linspace(0.0, 2*np.pi, num_angles)
        qc = self._quantum_circuit(backend,
                                   qubit_pairs,
                                   angles,
                                   cz_repetitions)

        super().__init__(backend=backend,
                        qubit_list=qubit_list,
                        qc=qc,
                        num_shots=num_shots,
                        use_ro_cal_points=use_ro_cal_points,
                        n_qubit_routine=2,
                        qubit_groups = qubit_pairs,
                        directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(qubit_pairs,
                            angles,
                            cz_repetitions)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        

    def _quantum_circuit(self,
                         backend,
                         qubit_pairs: list[list],
                         angles: np.array = np.linspace(0.0, 2*np.pi, 19),
                         cz_repetitions: int = 1,
                         cases: list = ['no_excitation', 'excitation']):
        """
        This instance method creates the experiment quantum circuit. For each
        variant (Off: 'no_excitation', On: 'excitation') and each phase phi, a
        measurement block resets the qubits, applies rx(pi/2) on q_targ (and rx(pi)
        on q_ctrl for On), applies the CZ gate(s), applies rz(phi) followed by
        rx(pi/2) on q_targ (and rx(pi) on q_ctrl for On), and measures all qubits
        simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_pairs (list(list)):
                An ordered list of qubit pairs [q_targ, q_ctrl] to be measured
                simultaneously, where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            angles (np.array):
                The phases phi of the recovery pi/2 pulse expressed in radians [rad].

            cz_repetitions (int):
                The number of consecutive CZ gates applied.

            cases (list):
                The experiment variants, 'no_excitation' (Off) and 'excitation' (On).
        """

        qc = QuantumCircuit(backend.num_qubits,
                            2*len(qubit_pairs)*len(angles)*len(cases),
                            name = f'Conditional_Oscillation_{backend.name}')

        bit_idx = 0
        for case in cases:
            for angle in angles:

                for qubit_pair in qubit_pairs:
                    for qubit_idx in qubit_pair:        
                        qc.reset(qubit_idx)
                qc.barrier()

                for qubit_pair in qubit_pairs:
                    if case == 'excitation':
                        qc.rx(np.pi, qubit_pair[1])
                    qc.rx(np.pi/2, qubit_pair[0])
                qc.barrier()

                for dummy_idx in range(cz_repetitions):
                    for qubit_pair in qubit_pairs:
                        qc.cz(qubit_pair[0], qubit_pair[1])
                    qc.barrier()

                for qubit_pair in qubit_pairs:
                    if case == 'excitation':
                        qc.rx(np.pi, qubit_pair[1])
                    qc.rz(angle, qubit_pair[0])
                    qc.rx(np.pi/2, qubit_pair[0])
                qc.barrier()

                for qubit_pair in qubit_pairs:
                    for qubit_idx in qubit_pair:
                        qc.measure(qubit = qubit_idx, cbit = bit_idx)
                        bit_idx += 1
                qc.barrier()
        
        return qc
    
    def _data_analysis(self,
                       qubit_pairs,
                       angles: np.array = np.linspace(0.0, 2*np.pi, 19),
                       cz_repetitions: int = 1):
        """
        This instance method fits the excited state population of q_targ in the
        Off and On variants with the cosine f(x) = a*cos(bx + c) + d (see cos_func
        in fitting_functions.py), and extracts for each qubit pair the phases, the
        conditional phase (phase difference), the oscillation offsets and amplitudes,
        and the missing fraction of q_ctrl. These are stored in the attribute
        'cond_osc_params', while the plots and the experiment data are stored in
        the project directory.

        Args:
            qubit_pairs (list(list)):
                An ordered list of qubit pairs [q_targ, q_ctrl] to be measured
                simultaneously, where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            angles (np.array):
                The phases phi of the recovery pi/2 pulse expressed in radians [rad].

            cz_repetitions (int):
                The number of consecutive CZ gates applied.
        """

        self.cond_osc_params = {f'Q{qubit_pairs[qubit_pair_idx]}': {} for qubit_pair_idx in range(len(qubit_pairs))}

        for qubit_pair_idx in range(len(qubit_pairs)):

            probs = self.result_probs_per_n_qubits[qubit_pair_idx]
            probs_per_qubit = self._collect_data_per_n_qubits(probs)

            q_targ_off = probs_per_qubit[0][0:int(len(probs_per_qubit[0])/2)]
            q_targ_on = probs_per_qubit[0][int(len(probs_per_qubit[0])/2):]
            q_ctrl_off = probs_per_qubit[1][0:int(len(probs_per_qubit[1])/2)]
            q_ctrl_on = probs_per_qubit[1][int(len(probs_per_qubit[1])/2):]

            q_targ_off_frac_in_1 = [q_targ_off[meas_idx]['1'] for meas_idx in range(len(q_targ_off))]
            q_targ_on_frac_in_1 = [q_targ_on[meas_idx]['1'] for meas_idx in range(len(q_targ_on))]
            q_ctrl_off_frac_in_1 = [q_ctrl_off[meas_idx]['1'] for meas_idx in range(len(q_ctrl_off))]
            q_ctrl_on_frac_in_1 = [q_ctrl_on[meas_idx]['1'] for meas_idx in range(len(q_ctrl_on))]

            angles_fit = np.linspace(0.0, 2*np.pi, 101)
            params_off, covariance_off = curve_fit(cos_func,
                                        angles,
                                        q_targ_off_frac_in_1,
                                        bounds=(
                                                    [0.1, 0.5, -np.inf, 0.1],   # lower bounds
                                                    [0.6, 1.2, np.inf, 0.9]  # upper bounds
                                            ))
            stderr_off = np.sqrt(np.diag(covariance_off))
            a_off_fit, b_off_fit, c_off_fit, d_off_fit = params_off
            cosine_off_fit = cos_func(angles_fit, a_off_fit, b_off_fit, c_off_fit, d_off_fit)

            params_on, covariance_on = curve_fit(cos_func,
                                        angles,
                                        q_targ_on_frac_in_1,
                                        bounds=(
                                                    [0.1, 0.5, -np.inf, 0.1],   # lower bounds
                                                    [0.6, 1.2, np.inf, 0.9]  # upper bounds
                                            ))
            stderr_on = np.sqrt(np.diag(covariance_on))
            a_on_fit, b_on_fit, c_on_fit, d_on_fit = params_on
            cosine_on_fit = cos_func(angles_fit, a_on_fit, b_on_fit, c_on_fit, d_on_fit)

            phase_off = ufloat(c_off_fit * (180/np.pi), stderr_off[2] * (180/np.pi)) % 360
            phase_on = ufloat(c_on_fit * (180/np.pi), stderr_on[2] * (180/np.pi)) % 360
            phase_diff = (phase_off - phase_on) % 360
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Phase Difference [deg]'] = phase_diff.nominal_value
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Phase Off [deg]'] = phase_off.nominal_value
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Phase On [deg]'] = phase_on.nominal_value

            osc_offs_off = ufloat(d_off_fit, stderr_off[3])
            osc_offs_on = ufloat(d_on_fit, stderr_on[3])
            offs_diff = abs(osc_offs_off-osc_offs_on)
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Oscillation offset difference [a.u.]'] = offs_diff.nominal_value
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Oscillation offset Off [a.u.]'] = osc_offs_off.nominal_value
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Oscillation offset On [a.u.]'] = osc_offs_on.nominal_value

            osc_amp_off = ufloat(a_off_fit, stderr_off[0])
            osc_amp_on = ufloat(a_on_fit, stderr_on[0])
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Oscillation amplitude Off'] = osc_amp_off.nominal_value
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Oscillation amplitude On'] = osc_amp_on.nominal_value

            fig, ax = plt.subplots(dpi=300)
            ax.plot(angles,
                    q_targ_off_frac_in_1,
                    alpha=0.9,
                    color='C0',
                    marker='o',
                    label = r'$q_{\text{ctrl}}$' + ' Off')
            ax.plot(angles,
                    q_targ_on_frac_in_1,
                    alpha=0.9,
                    color='C1',
                    marker='o',
                    label = r'$q_{\text{ctrl}}$' + ' On')
            ax.plot(angles_fit,
                    cosine_off_fit,
                    alpha=0.6,
                    color='C3',
                    linewidth=2.0,
                    label = 'Fit ' + r'$q_{\text{ctrl}}$' + ' Off')
            ax.plot(angles_fit,
                    cosine_on_fit,
                    alpha=0.6,
                    color='C4',
                    linewidth=2.0,
                    label = 'Fit ' + r'$q_{\text{ctrl}}$' + ' On')
            ax.axhline(osc_offs_off.nominal_value, color='C0', linestyle=':', alpha=0.9)
            ax.axhline(osc_offs_on.nominal_value, color='C1', linestyle=':', alpha=0.9)
            ax.set_xlabel('Phase (deg)')
            ax.set_ylabel(r'$q_{\text{targ.}}$'+' fraction in '+r'$|1\rangle$')
            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=f'Phase diff: {phase_diff:.1f} deg')
            ax.plot([], [], ' ', label=f'Phase Off: {phase_off:.1f} deg')
            ax.plot([], [], ' ', label=f'Phase On: {phase_on:.1f} deg')
            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=f'Offs. diff: {offs_diff*100:.2f} %')
            ax.plot([], [], ' ', label=f'Osc. offs. Off: {osc_offs_off:.4f}')
            ax.plot([], [], ' ', label=f'Osc. offs. On: {osc_offs_on:.4f}')
            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=f'Osc. amp. Off: {osc_amp_off:.4f}')
            ax.plot([], [], ' ', label=f'Osc. amp. On: {osc_amp_on:.4f}')
            deg_sign = r'$^{\circ}$'
            x_labels = ['0'+deg_sign, '60'+deg_sign, '120'+deg_sign, '180'+deg_sign,
                        '240'+deg_sign, '300'+deg_sign, '360'+deg_sign]
            label_locs = np.arange(0.0, 360+60, 60) * (np.pi/180)
            ax.set_xticks(label_locs)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_title(
                f'Conditional oscillation measurement'
                f'\n{self.backend_name} {self.backend_type}'
                f'\nQubit pair: Q{qubit_pairs[qubit_pair_idx]} || ' + r'$q_{\text{targ.}}=$' + \
                    f'Q{qubit_pairs[qubit_pair_idx][0]}, ' + r'$q_{\text{ctrl}}=$' + f'Q{qubit_pairs[qubit_pair_idx][1]}'
                f'\nCZ repetitions: {cz_repetitions}'
                f'\n{self.record.date_timestamp}_{self.record.job_timestamp}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            cond_osc_main_fig_path = (
                Path(self.record.project_dir)
                / f"cond_osc_Q{qubit_pairs[qubit_pair_idx]}_main_plot_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(cond_osc_main_fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            ctrl_off = ufloat(
                np.mean(q_ctrl_off_frac_in_1),
                sem(q_ctrl_off_frac_in_1)
            )
            ctrl_on = ufloat(
                np.mean(q_ctrl_on_frac_in_1),
                sem(q_ctrl_on_frac_in_1)
            )
            missing_fraction = ctrl_on - ctrl_off
            self.cond_osc_params[f'Q{qubit_pairs[qubit_pair_idx]}']['Missing fraction [%]'] = missing_fraction.nominal_value*100

            fig, ax = plt.subplots(dpi=300)
            ax.plot(angles,
                    q_ctrl_off_frac_in_1,
                    alpha=0.9,
                    color='C0',
                    marker='o',
                    label = r'$q_{\text{ctrl}}$' + ' Off')
            ax.plot(angles,
                    q_ctrl_on_frac_in_1,
                    alpha=0.9,
                    color='C1',
                    marker='o',
                    label = r'$q_{\text{ctrl}}$' + ' On')
            ax.set_xlabel('Phase (deg)')
            ax.set_ylabel(r'$q_{\text{ctrl}}$'+' fraction in '+r'$|1\rangle$')
            ax.plot([], [], ' ', label=' ')
            ax.plot([], [], ' ', label=f'Missing frac.: {missing_fraction*100:.1f} %')
            deg_sign = r'$^{\circ}$'
            x_labels = ['0'+deg_sign, '60'+deg_sign, '120'+deg_sign, '180'+deg_sign,
                        '240'+deg_sign, '300'+deg_sign, '360'+deg_sign]
            label_locs = np.arange(0.0, 360+60, 60) * (np.pi/180)
            ax.set_xticks(label_locs)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_title(
                f'Conditional oscillation measurement'
                f'\n{self.backend_name} {self.backend_type}'
                f'\nQubit pair: Q{qubit_pairs[qubit_pair_idx]} || ' + r'$q_{\text{targ.}}=$' + \
                    f'Q{qubit_pairs[qubit_pair_idx][0]}, ' + r'$q_{\text{ctrl}}=$' + f'Q{qubit_pairs[qubit_pair_idx][1]}'
                f'\nCZ repetitions: {cz_repetitions}'
                f'\n{self.record.date_timestamp}_{self.record.job_timestamp}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.8))
            cond_osc_spectator_fig_path = (
                Path(self.record.project_dir)
                / f"cond_osc_Q{qubit_pairs[qubit_pair_idx]}_spectator_plot_{self.record.date_timestamp}_{self.record.job_timestamp}.png"
            )
            fig.savefig(cond_osc_spectator_fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        self.experiment_data = {}
        self.experiment_data["Experiment name"] = self.record.project_name
        self.experiment_data["Experiment timestamp"] = f"{self.record.date_timestamp}_{self.record.job_timestamp}"
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit pairs"] = qubit_pairs
        self.experiment_data["Rotation angles [rad]"] = angles
        self.experiment_data["CZ repetitions"] = cz_repetitions
        self.experiment_data["Processed data"] = {f"Q{qubit_pairs[qubit_pair_idx]}":self.result_probs_per_n_qubits[qubit_pair_idx] \
                                                 for qubit_pair_idx in range(len(qubit_pairs))}
        self.experiment_data["Fitted data"] = self.cond_osc_params
        json_file_path = (
            Path(self.record.project_dir)
            / f"cond_osc_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)
    
    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        qubit_pairs = json_data['Qubit pairs']
        angles = json_data['Rotation angles [rad]']
        cz_repetitions = json_data['CZ repetitions']
        # cls.qubit_list = json_data['Qubit list']
        # cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls,
                           qubit_pairs,
                           angles,
                           cz_repetitions)
        return json_data
    
class BellStateMeasurement(BaseMeasurement):
    """
    This measurement class implements two-qubit state tomography of a Bell state,
    used for benchmarking the preparation of an entangled state on a qubit pair [1].

    One of the four Bell states (phi_plus, phi_minus, psi_plus, psi_minus) is
    prepared on each qubit pair, and both qubits are measured in each of the 9
    two-qubit Pauli bases {X, Y, Z} x {X, Y, Z}. From these measurements the
    expectation values of all 16 two-qubit Pauli operators are obtained, with
    the single-qubit terms (e.g. IZ) averaged over all bases in which they can
    be extracted, and <II> = 1 [2].

    Since a density matrix reconstructed directly from the measured expectation
    values is not guaranteed to be physical, the density matrix is obtained by
    a least-squares fit of the expectation values over physical density matrices
    rho = T^dagger T / Tr(T^dagger T), with T a lower triangular matrix, in the
    spirit of a maximum likelihood estimation [3]. The Bell-state fidelity is
    then given by F = <psi|rho|psi>, with |psi> the target Bell state.

    Further reading:
    [1] Entanglement and Quantum Error Correction with Superconducting Qubits,
        Matthew Reed, PhD Thesis, arXiv:1311.6759 (2013)
        https://doi.org/10.48550/arXiv.1311.6759
        Chapter 2, section 2.1.6.
    [2] Same as [1], Chapter 7, section 7.3.
    [3] Same as [1], Chapter 7, section 7.3.1.
    [4] PycQED, https://github.com/DiCarloLab-Delft/PycQED_py3
        HAL_Device.measure_two_qubit_tomo_bell()
    """

    def __init__(self,
                backend,
                qubit_pairs: list[list[int]],
                bell_state: str,
                num_shots: int,
                use_ro_cal_points: bool = True,
                directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_pairs (list(list)):
                An ordered list of qubit pairs to be measured simultaneously,
                where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            bell_state (str):
                The target Bell state, one of
                'phi_plus'  : (|00> + |11>)/sqrt(2),
                'phi_minus' : (|00> - |11>)/sqrt(2),
                'psi_plus'  : (|01> + |10>)/sqrt(2),
                'psi_minus' : (|01> - |10>)/sqrt(2).

            num_shots (int):
                The number of shots for which the experiment is executed.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """

        if bell_state not in ["phi_plus", "phi_minus",
                              "psi_plus", "psi_minus"]:
            raise ValueError("Invalid input bell_state. Available states are: "
                             "phi_plus, phi_minus, psi_plus, psi_minus.")
        
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
                        use_ro_cal_points=use_ro_cal_points,
                        n_qubit_routine=2,
                        qubit_groups = qubit_pairs,
                        directory=directory)
        self.project_name = self.record.project_name
        self.project_dir = self.record.project_dir
        self.date_timestamp = self.record.date_timestamp
        self.job_timestamp = self.record.job_timestamp
        
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Analyzing processed data ...")
        self._data_analysis(qubit_pairs,
                            bell_state)
        timestamp_now = datetime.now().replace(microsecond=0)
        print(f"[{timestamp_now}] "
              f"Data analysis complete!")
        

    def _quantum_circuit(self,
                        backend,
                        qubit_pairs: list[list[int]],
                        bell_state: str):
        """
        This instance method creates the experiment quantum circuit. For each of
        the 9 tomography bases, a measurement block resets the qubits, prepares the
        Bell state with single-qubit rotations followed by H and CNOT gates (which are
        decomposed into native gates, see BaseMeasurement), applies the
        pre-measurement rotations of the tomography basis (see apply_rotations_from_list
        in circuit_modifiers.py), and measures all qubits simultaneously.

        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            qubit_pairs (list(list)):
                An ordered list of qubit pairs to be measured simultaneously,
                where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            bell_state (str):
                The target Bell state, one of 'phi_plus', 'phi_minus',
                'psi_plus', or 'psi_minus'.
        """

        def create_bell_states(qc: QuantumCircuit,
                            qubit_pairs: list[list[int]],
                            bell_state: str):
            """
            Appends to qc the reset and Bell-state preparation of each qubit pair.
            """
            
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
                            name=f'Bell_State_Tomography_{backend.name}')
        
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
        """
        This instance method computes the expectation values of all 16 two-qubit
        Pauli operators for each qubit pair, reconstructs the density matrix (see
        _project_to_density_matrix) and computes the fidelity to the target Bell
        state. The fidelities are stored in the attribute 'bell_state_fidelities',
        while the plots and the experiment data are stored in the project directory.

        Args:
            qubit_pairs (list(list)):
                An ordered list of qubit pairs to be measured simultaneously,
                where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            bell_state (str):
                The target Bell state, one of 'phi_plus', 'phi_minus',
                'psi_plus', or 'psi_minus'.
        """

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
                tomography_dict[pauli_term] = observable_expectation_values_Z_basis([self.result_probs_per_n_qubits[qubit_pair_idx][pauli_term_idx]],
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
                        expectation_value = observable_expectation_values_Z_basis([self.result_probs_per_n_qubits[qubit_pair_idx][pauli_term_idx]],
                                                                                                sq_pauli_term_in_Z)[0]
                        tomography_value += expectation_value
                        num_expectation_values += 1
                tomography_dict[sq_pauli_term] = tomography_value / num_expectation_values

            tomography_dict = {pauli_term: tomography_dict[pauli_term] for pauli_term in pauli_labels}
            tomography_values = [tomography_dict[pauli_term] for pauli_term in tomography_dict.keys()]
            all_tomographies[f"Q{qubit_pairs[qubit_pair_idx]}"] = tomography_dict

            density_matrix = self._project_to_density_matrix(tomography_dict,
                                                                 pauli_labels)    
            density_matrix_real = np.real(density_matrix)
            density_matrix_imag = np.imag(density_matrix)      
            fidelity = float(np.real(np.vdot(ideal_bell_states[bell_state], density_matrix @ ideal_bell_states[bell_state])))
            self.bell_state_fidelities[f"Q{qubit_pairs[qubit_pair_idx]} [a.u.]"] = fidelity

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300)
            fig.subplots_adjust(top=0.75, wspace=0.3)
            fig.suptitle(
                f'Bell-state tomography\n{self.backend_name} {self.backend_type}\n'
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
        self.experiment_data["Backend name"] = self.backend_name
        self.experiment_data["Backend type"] = self.backend_type
        self.experiment_data["Number of shots"] = self.num_shots
        self.experiment_data["Qubit pairs"] = qubit_pairs
        self.experiment_data["Bell state"] = bell_state
        self.experiment_data["Processed data"] = {f"Q{qubit_pairs[qubit_pair_idx]}":self.result_probs_per_n_qubits[qubit_pair_idx] \
                                                 for qubit_pair_idx in range(len(qubit_pairs))}
        self.experiment_data["Tomography data"] = all_tomographies
        self.experiment_data["Bell-state fidelities"] = self.bell_state_fidelities
        json_file_path = (
            Path(self.record.project_dir)
            / f"tomography_data_{self.record.date_timestamp}_{self.record.job_timestamp}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(make_json_serializable(self.experiment_data), file, indent=3)

    def _project_to_density_matrix(self,
                                   tomography_dict: dict,
                                   pauli_labels: list):
        """
        This instance method reconstructs a physical two-qubit density matrix from
        the measured Pauli expectation values. The density matrix is parametrized as
        rho = T^dagger T / Tr(T^dagger T), with T a lower triangular matrix, which
        guarantees a Hermitian, positive semi-definite matrix of unit trace, and the
        16 real parameters of T are found by minimizing the sum of squared differences
        between the predicted Tr(rho P) and the measured expectation values <P>,
        starting from the identity matrix.

        Args:
            tomography_dict (dict):
                The measured expectation values of the two-qubit Pauli operators.
                e.g. tomography_dict = {'II': 1.0, 'IX': 0.01, ..., 'ZZ': 0.98}

            pauli_labels (list):
                The ordered list of the 16 two-qubit Pauli operator labels.
        """

        def build_density_matrix(t_params):
            """
            Returns the density matrix rho = T^dagger T / Tr(T^dagger T), with T the
            lower triangular matrix built from the 16 real parameters t_params.
            """
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
            """
            Returns the expectation values Tr(rho P) for each Pauli operator P.
            """
            return np.array([
                np.real(np.trace(rho @ P)) for P in pauli_ops
            ])

        def loss(params, measured_expectations, pauli_ops):
            """
            Returns the sum of squared differences between the predicted and the
            measured Pauli expectation values.
            """
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
    
    @classmethod
    def load_data(cls,
                  job_timestamp: str,
                  directory: str = None):
        """
        This class method loads the processed data of an already executed
        measurement from the local project record, and re-runs the data analysis,
        storing again the plots and the experiment data in the project directory.

        Args:
            job_timestamp (str):
                The experiment timestamp, as shown in the measurement plots.
                Should be given as "YYYYMMDD_HHMMSS".

            directory (str):
                Specifies the directory path in which the project record is
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".

        Returns:
            json_data (dict):
                The full experiment data, as stored in the JSON file.
        """

        json_data = super()._load_probs_per_n_qubits(cls,
                                                     job_timestamp=job_timestamp,
                                                     directory=directory)
        qubit_pairs = json_data['Qubit pairs']
        bell_state = json_data['Bell state']
        # cls.qubit_list = json_data['Qubit list']
        # cls.qubit_labels = [f"Q{qubit_idx}" for qubit_idx in cls.qubit_list]
        cls._data_analysis(cls,
                           qubit_pairs,
                           bell_state)
        return json_data

def make_json_serializable(obj):
    """
    This function recursively converts numpy arrays and numpy scalars, also when
    nested within dictionaries and lists, into native Python types, so that the
    experiment data can be stored in a JSON format.

    Args:
        obj:
            The object to be converted, e.g. the experiment data dictionary.
    """
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