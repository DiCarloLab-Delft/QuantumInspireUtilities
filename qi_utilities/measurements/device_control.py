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

    def measure_rabi(self,
                     qubit_list: list,
                     rotation_angles = np.linspace(0, 2*np.pi, num=29),
                     num_shots: int = None):
        
        if num_shots is not None:
            self.num_shots = num_shots
        rabi_meas = RabiMeasurement(backend=self.backend,
                                    qubit_list=qubit_list,
                                    rotation_angles=rotation_angles,
                                    num_shots=self.num_shots,
                                    directory=self.current_directory)
        self.latest_qc = rabi_meas.qc
        self.rabi_amplitudes = rabi_meas.rabi_amplitudes
        print(self.rabi_amplitudes)

    def measure_T1(self,
                   qubit_list: list,
                   measurement_times: np.array,
                   num_shots: int = None):

        if num_shots is not None:
            self.num_shots = num_shots
        T1_meas = T1_Measurement(backend=self.backend,
                                 qubit_list=qubit_list,
                                 measurement_times=measurement_times,
                                 num_shots=self.num_shots,
                                 directory=self.current_directory)
        self.latest_qc = T1_meas.qc
        self.T1_values = T1_meas.T1_values
        print(self.T1_values)

    def measure_T2_ramsey(self,
                          qubit_list: list,
                          measurement_times: np.array,
                          artificial_detuning: float = None,
                          num_shots: int = None):

        if num_shots is not None:
            self.num_shots = num_shots
        T2_ramsey_meas = T2_RamseyMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       artificial_detuning=artificial_detuning,
                                       num_shots=self.num_shots,
                                       directory=self.current_directory)
        self.latest_qc = T2_ramsey_meas.qc
        self.T2_ramsey_values = T2_ramsey_meas.T2_ramsey_values
        print(self.T2_ramsey_values)

    def measure_T2_echo(self,
                        qubit_list: list,
                        measurement_times: np.array,
                        artificial_detuning: float = None,
                        num_shots: int = None):

        if num_shots is not None:
            self.num_shots = num_shots
        T2_echo_meas = T2_EchoMeasurement(backend=self.backend,
                                       qubit_list=qubit_list,
                                       measurement_times=measurement_times,
                                       artificial_detuning=artificial_detuning,
                                       num_shots=self.num_shots,
                                       directory=self.current_directory)
        self.latest_qc = T2_echo_meas.qc
        self.T2_echo_values = T2_echo_meas.T2_echo_values
        print(self.T2_echo_values)

    def measure_allxy(self,
                      qubit_list: list,
                      num_shots: int = None):
        
        if num_shots is not None:
            self.num_shots = num_shots
        allxy_meas = AllXYMeasurement(backend=self.backend,
                                      qubit_list=qubit_list,
                                      num_shots=self.num_shots,
                                      directory=self.current_directory)
        self.latest_qc = allxy_meas.qc
        self.allxy_deviations = allxy_meas.allxy_deviations
        print(self.allxy_deviations)