import numpy as np
from qi_utilities.measurements.measurement_classes import *

class DeviceControl:

    def __init__(self,
                 backend,
                 directory: str = None):
        self.backend = backend
        self.current_directory = directory
        self.num_shots = 2**12
        self.latest_qc = None
        
        self.rabi_amplitudes = None
        self.T1_values = None
        self.T2_ramsey_values = None
        self.T2_echo_values = None
        self.allxy_deviations = None
        self.bell_state_fidelities = None

    def _check_qubit_list(self,
                          qubit_list):
        
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
        
        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        rabi_meas = RabiMeasurement(backend=self.backend,
                                    qubit_list=qubit_list,
                                    rotation_angles=rotation_angles,
                                    rotation_axis=rotation_axis,
                                    num_shots=self.num_shots,
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

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        T1_meas = T1_Measurement(backend=self.backend,
                                 qubit_list=qubit_list,
                                 measurement_times=measurement_times,
                                 num_shots=self.num_shots,
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

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        T2_ramsey_meas = T2_RamseyMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       num_shots=self.num_shots,
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

        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        T2_echo_meas = T2_EchoMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       num_shots=self.num_shots,
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
        
        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        flipping_meas = FlippingMeasurement(backend=self.backend,
                                            qubit_list=qubit_list,
                                            max_number_of_flips=max_number_of_flips,
                                            equator=equator,
                                            rotation_axis=rotation_axis,
                                            rotation_angle=rotation_angle,
                                            num_shots=self.num_shots,
                                            use_ro_cal_points=use_ro_cal_points,
                                            directory=self.current_directory)
        self.latest_qc = flipping_meas.qc
        self.flipping_parameters = flipping_meas.flipping_parameters
        print(self.flipping_parameters)

    def measure_allxy(self,
                      qubit_list: list | str,
                      num_shots: int = None,
                      use_ro_cal_points: bool = True):
        
        qubit_list = self._check_qubit_list(qubit_list)
        if num_shots is not None:
            self.num_shots = num_shots
        allxy_meas = AllXYMeasurement(backend=self.backend,
                                      qubit_list=qubit_list,
                                      num_shots=self.num_shots,
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
        
        if num_shots is not None:
            self.num_shots = num_shots
        cond_osc_meas = ConditionalOscMeasurement(backend=self.backend,
                                                  qubit_pairs=qubit_pairs,
                                                  num_angles=num_angles,
                                                  cz_repetitions=cz_repetitions,
                                                  num_shots=self.num_shots,
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
        
        if num_shots is not None:
            self.num_shots = num_shots
        fidelity_meas = BellStateMeasurement(backend=self.backend,
                                             qubit_pairs=qubit_pairs,
                                             bell_state=bell_state,
                                             num_shots=self.num_shots,
                                             use_ro_cal_points=use_ro_cal_points,
                                             directory=self.current_directory)
        self.latest_qc = fidelity_meas.qc
        self.bell_state_fidelities = fidelity_meas.bell_state_fidelities
        print(self.bell_state_fidelities)