import numpy as np
from qi_utilities.measurements.measurement_classes import *

class DeviceControl:

    def __init__(self,
                 backend):
        self.backend = backend
        self.num_shots = 2**12

        self.T1_values = None

    def measure_T1(self,
                   qubit_list: list,
                   measurement_times: np.array):

        T1_meas = T1_Measurement(backend=self.backend,
                                 qubit_list=qubit_list,
                                 measurement_times=measurement_times,
                                 num_shots=self.num_shots)
        self.T1_values = T1_meas.T1_values
        print(self.T1_values)

    def measure_rabi(self,
                     qubit_list: list,
                     rotation_angles = np.linspace(0, 2*np.pi, num=29)):
        
        rabi_meas = RabiMeasurement(backend=self.backend,
                                    qubit_list=qubit_list,
                                    rotation_angles=rotation_angles,
                                    num_shots=self.num_shots)