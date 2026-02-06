from qiskit_aer import AerSimulator
from qiskit import transpiler, transpile
import json
import os
from qi_utilities.device_simulation.noise_modelling import create_noise_model

class NoisySimulator(AerSimulator):
    
    def __init__(self,
                 backend_name,
                 model_noise = True):
        
        device_simulation_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(device_simulation_path, 'backend_parameters.json')
        with open(json_path, 'r') as file:
            simulator_specs = json.load(file)[backend_name]
            
        self.basis_gates = simulator_specs['Native operations']
        coupling_map = transpiler.CouplingMap(simulator_specs['Coupling map'])
        if model_noise == True:
            self.noise_model = create_noise_model(simulator_specs)
        else:
            self.noise_model = None
        super().__init__(n_qubits = simulator_specs['Qubit register'],
                         basis_gates = self.basis_gates,
                         coupling_map = coupling_map,
                         noise_model = self.noise_model)
        
        self.name = backend_name
        self.description = f'A custom C++ Qasm simulator of {backend_name}'
        self.options.shots = simulator_specs['Default shots']
        self.max_shots = simulator_specs['Max shots']
            
    def run(self, qc, shots, memory = False):
        
        # Force internal compilation according to simulator basis gates
        # and coupling map
        transpiled_qc = transpile(qc,
                                  self,
                                  coupling_map=self.coupling_map,
                                  basis_gates=self.basis_gates)
        
        return super().run(transpiled_qc,
                           noise_model=self.noise_model,
                           optimization_level = 0,
                           shots = shots,
                           memory = memory)