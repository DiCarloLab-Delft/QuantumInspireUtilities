import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from qi_utilities.utility_functions.midcircuit_msmt import obtain_binary_list, get_multi_qubit_counts



def split_raw_shots(result,
                    qubit_list: list,
                    circuit_nr: int = None):
    """
    Splits the raw shots into two groups for a circuit containing readout mitigation circuits
    at the end of it.
    """

    nr_qubits = len(qubit_list)
    raw_shots = result.get_memory(circuit_nr)

    experiment_shots = []
    ro_mitigation_shots = []

    for raw_shots_entry in range(len(raw_shots)):
        ro_mitigation_shots.append(raw_shots[raw_shots_entry][0:2**(nr_qubits+1)])
        experiment_shots.append(raw_shots[raw_shots_entry][2**(nr_qubits+1)::])
    return experiment_shots, ro_mitigation_shots


def ro_corrected_multi_qubit_prob(experiment_probs,
                                  ro_assignment_matrix,
                                  qubit_list: list):

    nr_qubits = len(qubit_list)
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
                                 qubit_list: list):

    nr_qubits = len(qubit_list)
    prepared_states = obtain_binary_list(nr_qubits)
    
    ro_counts_per_prepared_states = get_multi_qubit_counts(ro_mitigation_shots, nr_qubits)

    assignment_probability_matrix = np.zeros([len(prepared_states), len(prepared_states)], dtype=np.float_)

    for prepared_state_idx in range(len(prepared_states)):
        assignment_probability_matrix[prepared_state_idx] = np.array([ro_counts_per_prepared_states[prepared_state_idx][state]
                                                 for state in prepared_states]) / len(ro_mitigation_shots)

    return assignment_probability_matrix


def plot_ro_assignment_matrix(ro_assignment_matrix,
                              qubit_list: list):

    def red_white_green_cmap():
        # Number of samples for smoothness
        n = 256

        # Fractions of the full range
        reds_frac   = 20 / 100
        middle_frac = 50 / 100 - reds_frac

        reds_n   = int(n * reds_frac)
        middle_n = int(n * middle_frac)
        greens_n = n - reds_n - middle_n

        reds = plt.cm.Reds(np.linspace(0.0, 1.0, reds_n))
        middle = np.ones((middle_n, 4))  # white
        greens = plt.cm.Greens(np.linspace(0.0, 1.0, greens_n))

        colors = np.vstack((reds, middle, greens))
        return LinearSegmentedColormap.from_list("RedWhiteGreen", colors)

    nr_qubits = len(qubit_list)
    binary_labels = []
    for binary_str_idx in range(2**nr_qubits):
        binary_labels.append(r"$|$" + f"{np.binary_repr(binary_str_idx, nr_qubits)}" + r"$\rangle$")

    fig_size = max(6, len(binary_labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    qubit_list_label = r"$|$"
    for qubit_idx in qubit_list[::-1]:
        qubit_list_label += f" Q{qubit_idx} "
    qubit_list_label += r"$\rangle$"

    ax.set_xticks(np.arange(len(binary_labels)))
    ax.set_yticks(np.arange(len(binary_labels)))
    ax.set_xticklabels(binary_labels)
    ax.set_yticklabels(binary_labels)

    base_fontsize = max(3, 14 - 0.2 * len(binary_labels))
    ax.tick_params(axis="x", labelsize=base_fontsize)
    ax.tick_params(axis="y", labelsize=base_fontsize)
    ax.tick_params(axis='x', rotation=60)

    ax.xaxis.tick_top()
    ax.set_xlabel("Declared state")
    ax.xaxis.set_label_position('top')
    ax.set_ylabel("Prepared state")
    ax.set_title(f"Readout assignment matrix\nQubit list: {qubit_list_label}")

    plt.setp(ax.get_xticklabels(), ha="center")

    values = np.zeros((len(binary_labels), len(binary_labels)))
    for i in range(len(binary_labels)):
        for j in range(len(binary_labels)):
            values[i, j] = ro_assignment_matrix[i, j] * 100
            cell_fontsize = max(2, 10 - 0.15 * len(binary_labels))
            txt = ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=cell_fontsize)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal()
            ])

    cmap = red_white_green_cmap()
    norm = Normalize(vmin=0, vmax=100)

    cax = ax.imshow(values, cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Probability (%)")
    cbar.set_ticks(ticks=list(np.arange(0, 110, 10)))

    plt.tight_layout()
    plt.show()
