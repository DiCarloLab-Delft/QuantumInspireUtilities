"""
Utility classes for creating a general (noisy) simulator using
the Qiskit AerSimulator class.

The AerJob is modified in such a way so that the returned results
have a simular structure to that of QIJob of the Quantum Inspire SDK.

Authors: Marios Samiotis, Jan Hemink
"""

import os
import json
import uuid
from datetime import datetime
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpiler, transpile
from qiskit.circuit import Delay
from qiskit.circuit import CircuitInstruction
from qiskit_aer import AerSimulator, AerJob
from qi_utilities.device_simulation.noise_modelling import create_noise_model

@dataclass
class job_result_data:
    job_id: uuid.UUID
    created_on: datetime
    shots_requested: int
    shots_done: int
    results: dict[str, int]
    raw_data: list | None = None
    execution_time_in_seconds: float = 0.0
    id: str = ''

@dataclass
class circuit_run_data:
    circuit: QuantumCircuit
    job_id: uuid.UUID
    results: job_result_data

class SimulatorJob(AerJob):
    def __init__(self, backend, job_id, fn, circuits=None, parameter_binds=None, run_options=None, executor=None):
        self.program_name = circuits[0].name
        super().__init__(backend, job_id, fn, circuits, parameter_binds, run_options, executor)

    def result(self, timeout=None):
        result = super().result(timeout)
        self.circuits_run_data = [
            circuit_run_data(
                circuit = circuit,
                job_id=self.job_id(),
                results = job_result_data(
                    created_on=datetime.now(),
                    job_id=self.job_id(),
                    shots_requested=self._run_options['shots'],
                    shots_done=self._run_options['shots'],
                    results=result.get_counts(idx),
                    raw_data=result.get_memory(idx) if self._run_options.get('memory',False) else None,
                )
            )
            for idx, circuit in enumerate(self.circuits())
        ]
        return result

class NoisySimulator(AerSimulator):
    
    def __init__(self,
                 backend_name,
                 ideal_simulation = False):
        
        device_simulation_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(device_simulation_path, 'backend_parameters.json')
        with open(json_path, 'r') as file:
            simulator_specs = json.load(file)[backend_name]
            
        self.basis_gates = simulator_specs['Native operations']
        coupling_map = transpiler.CouplingMap(simulator_specs['Coupling map'])
        if ideal_simulation == False:
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

    def unpack_qc_delays(self,
                         qc: QuantumCircuit):

        qc_new = QuantumCircuit(qc.num_qubits,
                                qc.num_clbits,
                                name = qc.name)

        for instruction_idx in range(len(qc)):
            if qc[instruction_idx].operation.name == 'delay':
                for repetition in range(qc[instruction_idx].operation.duration):
                    unit_delay_operation = Delay(duration = 1)
                    unit_delay_instruction = CircuitInstruction(operation=unit_delay_operation,
                                                                qubits=qc[instruction_idx].qubits)
                    qc_new.append(unit_delay_instruction)
            else:
                qc_new.append(qc[instruction_idx])

        return qc_new
            
    def run(self,
            qc,
            shots,
            memory = False):
        
        # Force internal compilation according to simulator basis gates
        # and coupling map
        if type(qc) == list:
            transpiled_qc = []
            for qc_idx in range(len(qc)):
                single_transpiled_qc = transpile(qc[qc_idx],
                                        backend = self,
                                        layout_method = "trivial",
                                        routing_method = "none",
                                        optimization_level = 0,
                                        basis_gates = self.basis_gates)
                single_transpiled_qc = self.unpack_qc_delays(single_transpiled_qc) # so that noise during delay is applied correctly
                transpiled_qc.append(single_transpiled_qc)
        else:
            transpiled_qc = transpile(qc,
                                      backend = self,
                                      layout_method = "trivial",
                                      routing_method = "none",
                                      optimization_level = 0,
                                      basis_gates = self.basis_gates)
            transpiled_qc = self.unpack_qc_delays(transpiled_qc) # so that noise during delay is applied correctly
        
        return super().run(transpiled_qc,
                           noise_model=self.noise_model,
                           optimization_level = 0,
                           shots = shots,
                           memory = memory)
    
    def _run_circuits(self, circuits, parameter_binds, **run_options):
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = SimulatorJob(
            self,
            job_id,
            self._execute_circuits_job,
            parameter_binds=parameter_binds,
            circuits=circuits,
            run_options=run_options,
        )
        aer_job.submit()

        return aer_job