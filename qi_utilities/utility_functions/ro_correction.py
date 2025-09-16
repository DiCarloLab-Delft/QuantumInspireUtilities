import numpy as np
from scipy.optimize import minimize
from midcircuit_msmt import obtain_binary_list


def split_raw_shots(result,
                    nr_qubits: int,
                    circuit_nr: int = None):
    """
    Splits the raw shots into two groups for a circuit containing readout mitigation circuits
    at the end of it.
    """

    raw_shots = result.get_memory(circuit_nr)

    experiment_shots = []
    ro_mitigation_shots = []

    for raw_shots_entry in range(len(raw_shots)):
        ro_mitigation_shots.append(raw_shots[raw_shots_entry][0:2**(nr_qubits+1)])
        experiment_shots.append(raw_shots[raw_shots_entry][2**(nr_qubits+1)::])
    return experiment_shots, ro_mitigation_shots


def ro_corrected_multi_qubit_prob(experiment_probs,
                                  ro_assignment_matrix,
                                  nr_qubits: int):

    binary_list = obtain_binary_list(nr_qubits)
    
    experiment_probs_ro_corrected = []

    for entry_idx in range(len(experiment_probs)):

        probs = np.array(list(experiment_probs[entry_idx].values()))

        def objective(x):
            return np.linalg.norm(ro_assignment_matrix @ x - probs) ** 2
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1
        }
        bounds = [(0, 1)] * len(probs)
        result = minimize(
            objective,
            x0=np.ones(len(probs)) / len(probs),  # initial guess: uniform distribution
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        probs_ro_corrected_dict = {}
        for idx in range(len(result.x)):
            probs_ro_corrected_dict[binary_list[idx]] = result.x[idx]

        experiment_probs_ro_corrected.append(probs_ro_corrected_dict)
    return experiment_probs_ro_corrected


def extract_ro_assignment_matrix(ro_mitigation_shots,
                                 nr_qubits: int):

    binary_list = obtain_binary_list(nr_qubits)
    
    ro_counts = get_multi_qubit_counts(ro_mitigation_shots, nr_qubits)

    declared_states_per_prepared_states = {}
    for binary_str_idx in range(len(binary_list)):
        declared_states_per_prepared_states[binary_list[binary_str_idx]] = ro_counts[binary_str_idx]

    # below is Tim's code
    assignment_matrix = np.empty([len(binary_list), len(binary_list)], dtype=np.float_)
    for i, (prepared_state, counts_per_declared_state) in enumerate(declared_states_per_prepared_states.items()):
        assignment_matrix[i, :] = list(counts_per_declared_state.values())
    assignment_probability_matrix = assignment_matrix / len(ro_mitigation_shots)
    # Tim's code ends here
    return assignment_probability_matrix