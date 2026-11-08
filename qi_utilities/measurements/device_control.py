"""
Module implementing the "DeviceControl" class, a user-friendly interface
for running the characterization experiments of measurement_classes.py on
a backend, and for keeping track of their latest results.

Authors: Marios Samiotis
"""

import numpy as np
from qi_utilities.measurements.measurement_classes import *

class DeviceControl:
    """
    This class serves as a single entry point for benchmarking a backend. Each
    "measure_" instance method executes the corresponding measurement class
    (see measurement_classes.py), prints its main result, and stores it as an
    attribute of the class, so that the latest result of each experiment is
    always accessible.

    e.g.
        device = DeviceControl(backend)
        device.measure_T1(qubit_list='all')
        device.T1_values # the latest measured T1 values
        device.latest_qc.draw('mpl') # the latest executed quantum circuit
    """

    def __init__(self,
                 backend,
                 directory: str = None):
        """
        Args:
            backend (QIBackend | NoisySimulator):
                The hardware or simulator backend.

            directory (str):
                Specifies the directory path in which the project record is to be
                stored.
                For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        self.backend = backend
        self.current_directory = directory
        self.num_shots = 2**12
        self.latest_qc = None
        
        self.rabi_amplitudes = None
        self.T1_values = None
        self.T2_ramsey_values = None
        self.T2_echo_values = None
        self.flipping_parameters = None
        self.allxy_deviations = None
        self.cond_osc_params = None
        self.bell_state_fidelities = None

    def _check_qubit_list(self,
                          qubit_list):
        """
        This instance method checks the input qubit_list, and converts the
        entry 'all' to the list of all qubit indices of the backend.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.
        """

        if type(qubit_list) == str:
            if qubit_list != 'all':
                raise ValueError("Invalid input qubit_list: it can either be an array of qubit indices or 'all'.")
            else:
                qubit_list = [qubit_idx for qubit_idx in range(self.backend.num_qubits)]
        return qubit_list

    def measure_rabi(self,
                     qubit_list: list | str,
                     rotation_angles: np.array = np.linspace(0, 2*np.pi, num=29),
                     rotation_axis: str = 'x',
                     num_shots: int = None,
                     use_ro_cal_points: bool = True):
        """
        This instance method performs a Rabi oscillation (see RabiMeasurement)
        and stores the result in the attribute 'rabi_amplitudes'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

            rotation_angles (np.array):
                The different rotation angles for the Rabi oscillation
                expressed in radians [rad].

            rotation_axis (str):
                The rotation axis of the Bloch sphere, either 'x' or 'y'.

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        rabi_meas = RabiMeasurement(backend=self.backend,
                                    qubit_list=qubit_list,
                                    rotation_angles=rotation_angles,
                                    rotation_axis=rotation_axis,
                                    num_shots=num_shots,
                                    use_ro_cal_points=use_ro_cal_points,
                                    directory=self.current_directory)
        self.latest_qc = rabi_meas.qc
        self.rabi_amplitudes = rabi_meas.rabi_amplitudes
        print(self.rabi_amplitudes)

    def measure_T1(self,
                   qubit_list: list | str,
                   measurement_times: np.array = np.linspace(0, 150e-6, num=41),
                   num_shots: int = None,
                   use_ro_cal_points: bool = True):
        """
        This instance method performs a T1 measurement (see T1_Measurement)
        and stores the result in the attribute 'T1_values'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        T1_meas = T1_Measurement(backend=self.backend,
                                 qubit_list=qubit_list,
                                 measurement_times=measurement_times,
                                 num_shots=num_shots,
                                 use_ro_cal_points=use_ro_cal_points,
                                 directory=self.current_directory)
        self.latest_qc = T1_meas.qc
        self.T1_values = T1_meas.T1_values
        print(self.T1_values)

    def measure_T2_ramsey(self,
                          qubit_list: list | str,
                          measurement_times: np.array,
                          num_shots: int = None,
                          use_ro_cal_points: bool = True):
        """
        This instance method performs a Ramsey experiment (see T2_RamseyMeasurement)
        and stores the result in the attribute 'T2_ramsey_values'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

            measurement_times (np.array):
                The measurement times expressed in units of seconds [s].

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        T2_ramsey_meas = T2_RamseyMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       num_shots=num_shots,
                                       use_ro_cal_points=use_ro_cal_points,
                                       directory=self.current_directory)
        self.latest_qc = T2_ramsey_meas.qc
        self.T2_ramsey_values = T2_ramsey_meas.T2_ramsey_values
        print(self.T2_ramsey_values)

    def measure_T2_echo(self,
                        qubit_list: list | str,
                        measurement_times: np.array,
                        num_shots: int = None,
                        use_ro_cal_points: bool = True):
        """
        This instance method performs a Hahn echo experiment (see T2_EchoMeasurement)
        and stores the result in the attribute 'T2_echo_values'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

            measurement_times (np.array):
                The total free evolution times (between the two pi/2 pulses)
                expressed in units of seconds [s].

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        T2_echo_meas = T2_EchoMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       num_shots=num_shots,
                                       use_ro_cal_points=use_ro_cal_points,
                                       directory=self.current_directory)
        self.latest_qc = T2_echo_meas.qc
        self.T2_echo_values = T2_echo_meas.T2_echo_values
        print(self.T2_echo_values)

    def measure_flipping(self,
                         qubit_list: list | str,
                         max_number_of_flips: int = 30,
                         equator: bool = True,
                         rotation_axis: str = 'x', # 'x' or 'y'
                         rotation_angle: str = '180',
                         num_shots: int = None,
                         use_ro_cal_points: bool = True): # '180' or '90'
        """
        This instance method performs a flipping experiment (see FlippingMeasurement)
        and stores the result in the attribute 'flipping_parameters'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

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

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        flipping_meas = FlippingMeasurement(backend=self.backend,
                                            qubit_list=qubit_list,
                                            max_number_of_flips=max_number_of_flips,
                                            equator=equator,
                                            rotation_axis=rotation_axis,
                                            rotation_angle=rotation_angle,
                                            num_shots=num_shots,
                                            use_ro_cal_points=use_ro_cal_points,
                                            directory=self.current_directory)
        self.latest_qc = flipping_meas.qc
        self.flipping_parameters = flipping_meas.flipping_parameters
        print(self.flipping_parameters)

    def measure_allxy(self,
                      qubit_list: list | str,
                      num_shots: int = None,
                      use_ro_cal_points: bool = True):
        """
        This instance method performs an AllXY experiment (see AllXYMeasurement)
        and stores the result in the attribute 'allxy_deviations'.

        Args:
            qubit_list (list | str):
                An ordered list containing the integer indices of the qubits
                used in the original quantum circuit.
                e.g. for qubits q0 and q2, qubit_list = [0, 2].
                Alternatively, qubit_list = 'all' for all qubits of the backend.

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is None:
            num_shots = self.num_shots
        allxy_meas = AllXYMeasurement(backend=self.backend,
                                      qubit_list=qubit_list,
                                      num_shots=num_shots,
                                      use_ro_cal_points=use_ro_cal_points,
                                      directory=self.current_directory)
        self.latest_qc = allxy_meas.qc
        self.allxy_deviations = allxy_meas.allxy_deviations
        print(self.allxy_deviations)

    def measure_conditional_oscillation(self,
                                        qubit_pairs: list[list],
                                        num_angles: int = 19,
                                        cz_repetitions: int = 1,
                                        num_shots: int = None,
                                        use_ro_cal_points: bool = True):
        """
        This instance method performs a conditional oscillation experiment
        (see ConditionalOscMeasurement) and stores the result in the attribute
        'cond_osc_params'.

        Args:
            qubit_pairs (list(list)):
                An ordered list of qubit pairs [q_targ, q_ctrl] to be measured
                simultaneously, where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            num_angles (int):
                The number of phases phi of the recovery pi/2 pulse, equally
                spaced in [0, 2*pi].

            cz_repetitions (int):
                The number of consecutive CZ gates applied.

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        if num_shots is None:
            num_shots = self.num_shots
        cond_osc_meas = ConditionalOscMeasurement(backend=self.backend,
                                                  qubit_pairs=qubit_pairs,
                                                  num_angles=num_angles,
                                                  cz_repetitions=cz_repetitions,
                                                  num_shots=num_shots,
                                                  use_ro_cal_points=use_ro_cal_points,
                                                  directory=self.current_directory)
        self.latest_qc = cond_osc_meas.qc
        self.cond_osc_params = cond_osc_meas.cond_osc_params
        print(self.cond_osc_params)

    def measure_bell_state_fidelity(self,
                                    qubit_pairs: list[list],
                                    bell_state: str,
                                    num_shots: int = None,
                                    use_ro_cal_points: bool = True):
        """
        This instance method performs two-qubit state tomography of a Bell state
        (see BellStateMeasurement) and stores the result in the attribute
        'bell_state_fidelities'.

        Args:
            qubit_pairs (list(list)):
                An ordered list of qubit pairs to be measured simultaneously,
                where each pair must be coupled by a CZ gate.
                A single qubit cannot be used in multiple pairs.
                e.g. qubit_pairs = [[0, 2], [1, 3]].

            bell_state (str):
                The target Bell state, one of 'phi_plus', 'phi_minus',
                'psi_plus', or 'psi_minus'.

            num_shots (int):
                The number of shots for which the experiment is executed.
                Defaults to the attribute 'num_shots'.

            use_ro_cal_points (bool):
                If True, readout errors are mitigated using the readout
                calibration points.
        """

        if num_shots is None:
            num_shots = self.num_shots
        fidelity_meas = BellStateMeasurement(backend=self.backend,
                                             qubit_pairs=qubit_pairs,
                                             bell_state=bell_state,
                                             num_shots=num_shots,
                                             use_ro_cal_points=use_ro_cal_points,
                                             directory=self.current_directory)
        self.latest_qc = fidelity_meas.qc
        self.bell_state_fidelities = fidelity_meas.bell_state_fidelities
        print(self.bell_state_fidelities)