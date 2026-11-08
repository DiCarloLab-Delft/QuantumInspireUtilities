"""
Utility functions for noise learning.

Authors: Jan Hemink
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, overload

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, SdgGate, SGate
from qiskit.quantum_info import Pauli
from scipy.optimize import curve_fit, minimize

from qi_utilities.error_mitigation.toolkit.helper_functions import _check_pauli_str, paulis_commute
from qi_utilities.error_mitigation.toolkit.plotting import plot_multi_bar, set_colour_cycle_10

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from qiskit.circuit import Gate


MODEL_PAULIS = [
    "XI", "YI", "ZI",
    "IX", "IY", "IZ",
    "XX", "XY", "XZ",
    "YX", "YY", "YZ",
    "ZX", "ZY", "ZZ",
]
CZ_DURATION = 60e-9  # would be nice to be able to get this from the backend

H = HGate()
Sdg = SdgGate()
S = SGate()
BASIS_GATES = {
    "X": [H],
    "Y": [H, S],
}


def _gates_for_basis(pauli: str) -> list[Gate]:
    return BASIS_GATES.get(pauli.upper(), [])


CZ_BASIS_CORRECTION_GATES = {
    "X": [S],
    "Y": [Sdg],
}


def _gates_for_correction(basis: str) -> list[Gate]:
    return CZ_BASIS_CORRECTION_GATES.get(basis.upper(), [])


MEASURED_FIDELITIES_PER_BASIS = {
    "ZZ": {"IZ": "IZ-IZ", "ZI": "ZI-ZI", "ZZ": "ZZ-ZZ"},
    "XX": {"IX": "IX-ZY", "XI": "XI-YZ", "XX": "XX-XX"},
    "YY": {"IY": "IY-ZX", "YI": "YI-XZ", "YY": "YY-YY"},
    "XY": {"IY": "IY-ZX", "XI": "XI-YZ", "XY": "XY-XY"},
    "YX": {"IX": "IX-ZY", "YI": "YI-XZ", "YX": "YX-YX"},
}
MEASURED_FIDELITIES_PER_BASIS_NO_ROTATIONS = {
    "ZZ": {"IZ": "IZ-IZ", "ZI": "ZI-ZI", "ZZ": "ZZ-ZZ"},
    "XX": {"IX": "IX-ZX", "XI": "XI-XZ", "XX": "XX-YY"},
    "YY": {"IY": "IY-ZY", "YI": "YI-YZ", "YY": "XX-YY"},
    "XY": {"IY": "IY-ZY", "XI": "XI-XZ", "XY": "XY-YX"},
    "YX": {"IX": "IX-ZX", "YI": "YI-YZ", "YX": "XY-YX"},
}


def _measured_fidelities_per_basis(
    basis: str,
    single_qubit_rotations: bool = True,
) -> dict[str, str]:
    if single_qubit_rotations:
        return MEASURED_FIDELITIES_PER_BASIS[basis.upper()]
    return MEASURED_FIDELITIES_PER_BASIS_NO_ROTATIONS[basis.upper()]


def _fidelity_measurement_circuit(
    basis: str,
    k: int,
    single_qubit_rotations: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Actual implementation that constructs the circuits to measure the Pauli fidelities.
    See `cz_pauli_fidelity_measurement_circuit` for more information.
    """
    if name is None:
        name = f"{basis}-basis_Pauli_fidelities"

    num_qubits = len(basis)
    qubit_list = list(range(num_qubits))
    qubit_pairs = [(i, i + 1) for i in range(0, num_qubits, 2)]

    qc = QuantumCircuit(num_qubits, num_qubits, name=name)

    # prepare qubit 0 state
    qc.reset(qubit_list)
    qc.barrier()

    # convert to basis
    for qubit_idx, pauli in enumerate(reversed(basis)): # reverse basis as python iterates left to right over strings
        for gate in _gates_for_basis(pauli):
            qc.append(gate, (qc.qubits[qubit_idx],))

    if k == 0:
        qc.barrier()

    # apply repeated CZ gates
    for _ in range(k // 2):
        for pair in qubit_pairs:
            qc.cz(*pair)

        if single_qubit_rotations:
            # depending on the basis we can do a correction to get a single fidelity squared instead of a fidelity product
            for qubit_idx, pauli in enumerate(reversed(basis)):
                for gate in _gates_for_correction(pauli):
                    qc.append(gate, (qc.qubits[qubit_idx],))

        for pair in qubit_pairs:
            qc.cz(*pair)

        if single_qubit_rotations:
            for qubit_idx, pauli in enumerate(reversed(basis)):
                for gate in reversed(_gates_for_correction(pauli)):
                    qc.append(gate.inverse(), (qc.qubits[qubit_idx],))

    # rotate back to Z basis for measurement
    for qubit_idx, pauli in enumerate(reversed(basis)):
        for gate in reversed(_gates_for_basis(pauli)):
            qc.append(gate.inverse(), (qc.qubits[qubit_idx],))

    qc.barrier()

    # do measurements
    for idx in range(num_qubits):
        qc.measure(idx, idx)

    qc.barrier()
    return qc


def cz_pauli_fidelity_measurement_circuit(
    basis: str,
    k: int,
    single_qubit_rotations: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Constructs a circuit to measure the fidelities of the Pauli noise channel affecting the CZ gate.

    IMPORTANT: resulting circuit still needs to be Pauli twirled (to ensure a Pauli noise channel),
    and also still needs to be transpiled to assign the physical qubits that will be used (and
    reduce the number of single qubit gates).

    Post-initialisation and pre-measurement rotations are applied such that the expectation value
    is measured in the provided (Pauli) basis. The circuit is constructed using `k` repetitions of
    the CZ gate, `k` should be divisible by 2 to ensure that the circuit is measured in the same
    basis as initialised. Measurement of the circuit yields a product of two Pauli fidelities.

    Single qubit rotations (`CZ_BASIS_CORRECTION_GATES`) can be applied after each CZ gate to ensure
    that when the 'XX', 'YY','XY', and 'YX' fidelities are measured they form a product with
    themselves such that `f^k` is measured as opposed to `(f_1 * f_2)^k/2`.

    Parameters
    ----------
    basis : str
        2-qubit Pauli string corresponding to the basis that will be used for executing the circuit.
        Will apply post-initialisation and pre-measurement rotations to ensure that the
        expectation value will be measured in this basis. 'XY' will have 'Y' basis on first
        qubit and 'X' basis on second qubit.
    k : int
        The number of CZ repetitions, should be divisible by 2.
    single_qubit_rotations : bool, optional
        If True, applies single qubit rotations after each CZ gate such that 'XX', 'YY', 'XY', and
        'YX', fidelities are measured as a pair with themselves (`f^k` as opposed to `(f_1 * f_2)^k/2`)
    name : str | None, optional
        Name of the `QuantumCircuit`,
        when not provided will default to `"{basis}-basis_Pauli_fidelities_k={k}"`.

    Returns
    -------
    QuantumCircuit
        Used to measure Pauli fidelities. Still needs to be Pauli twirled and transpiled.

    See Also
    --------
    `_fidelity_measurement_circuit` : For the actual implementation of constructing the circuit.
    `pauli_twirling.pauli_twirl_cz` : For Pauli twirling the resulting circuit.
    """
    if len(basis) % 2 != 0:
        raise ValueError(
            "Basis does not have length 2, corresponding to the basis of the first and second qubits."
        )
    _check_pauli_str(basis)

    if k % 2 != 0:
        raise ValueError("`k` should be divisible by 2.")

    return _fidelity_measurement_circuit(basis, k, single_qubit_rotations, name)


@overload
def _exponential(k: float, a: float, f: float, b: float) -> float: ...
@overload
def _exponential(k: NDArray, a: float, f: float, b: float) -> NDArray: ...


def _exponential(k: float | NDArray, a: float, f: float, b: float) -> float | NDArray:
    """
    Exponential used for fitting in the Pauli fidelity measurement experiment.
    Fits the function:  `a*f^(k/2)+b`, where f is a fidelity pair.
    """
    return a * f ** (k / 2) + b


def fit_fidelity_pairs(
    expectation_values: dict[str, dict[str, dict[str, float]]],
    k_values: list[int],
) -> tuple[dict[str, dict[str, float]], dict]:
    fidelity_pairs: dict[str, dict[str, float]] = {}
    fit_results = {}

    for qubit_pair, data in expectation_values.items():
        fidelity_pairs[qubit_pair] = {}
        fit_params = {}
        fit_uncertainty = {}
        sum_squared_residual = {}
        sum_squares_total = {}
        r_square = {}

        # when a fidelity pair is measured in multiple bases, average the measured data points before fitting and fit only once.
        data_points = {}
        k_values = np.array(k_values)  # pyright: ignore[reportAssignmentType]
        for basis, pair_dict in data.items():
            for pair, measured_data in pair_dict.items():
                if pair not in data_points:
                    data_points[pair] = []
                data_points[pair].append(measured_data)

        for pair in data_points:
            data_points[pair] = np.mean(data_points[pair], axis=0)

            # perform fitting
            bounds = ([0.01, 0.5, 0], [1, 1, np.max(data_points[pair])])
            params, cov = curve_fit(
                f=_exponential,
                xdata=k_values,
                ydata=data_points[pair],
                p0=[1, 1, 0],
                bounds=bounds,
                maxfev=10000,
            )
            # analyse fit results
            uncertainties = np.sqrt(np.diag(cov)).tolist()
            fit_params[pair] = {key: params[idx] for idx, key in enumerate(["a", "f1f2", "b"])}
            fit_uncertainty[pair] = {key: uncertainties[idx] for idx, key in enumerate(["a", "f1f2", "b"])}
            sum_squared_residual[pair] = np.sum((data_points[pair] - _exponential(k_values, *params))**2)  # type: ignore
            sum_squares_total[pair] = np.sum((data_points[pair] - data_points[pair].mean() )**2)
            r_square[pair] = 1 - sum_squared_residual[pair] / sum_squares_total[pair]

            fidelity_pairs[qubit_pair][pair] = fit_params[pair]["f1f2"]

        # compile fit results per qubit pair
        fit_results[qubit_pair] = {
            "Fit parameters": fit_params,
            "Fit uncertainty": fit_uncertainty,
            "SSR": sum_squared_residual,
            "SST": sum_squares_total,
            "R2": r_square,
            "Combined expectation values": data_points,
        }

    return fidelity_pairs, fit_results


def matrix_m(
    benchmark_paulis: Sequence[str],
    model_paulis: Sequence[str],
) -> NDArray:
    """
    Constructs binary matrix M based on commutation between lists of Pauli strings.

    The rows correspond to the benchmark Pauli's and the columns to the model Pauli's.
    The matrix entries are either 0 when the benchmark and model Pauli commute, or 1 when they do not.

    Parameters
    ----------
    benchmark_paulis : list[str]
        List containing all Pauli strings of the benchmark Pauli's. The rows of the matrix will
        correspond to these.
    model_paulis : list[str]
        List containing all Pauli strings of the model Pauli's. The columns of the matrix will
        correspond to these.

    Returns
    -------
    NDArray
        Binary matrix M, used for learning Pauli noise channels.
    """
    num_cols = len(model_paulis)
    num_rows = len(benchmark_paulis)
    matrix = np.empty((num_rows, num_cols), int)
    for i in range(num_rows):
        for j in range(num_cols):
            matrix[i, j] = not paulis_commute(benchmark_paulis[i], model_paulis[j])
    return matrix


def _solve_nnls(
    M: NDArray,
    f: Sequence,
    method: str = "SLSQP",
    non_negative: bool = True,
) -> dict[str, float]:
    """Solves Mx = -log(f)/(2t) for x."""

    def objective(x):
        return np.linalg.norm(M @ x + np.log(f) / (2 * CZ_DURATION)) ** 2

    bounds = [(0, None)] * len(MODEL_PAULIS) if non_negative else None
    result = minimize(
        objective,
        x0=np.ones(len(MODEL_PAULIS)) * 1000,
        bounds=bounds,
        method=method,
        tol=1e-16,
    )
    return dict(zip(MODEL_PAULIS, result.x))


def _extract_pauli_rates_from_pairs(fidelity_pairs: dict[str, float], non_negative: bool = True) -> dict[str, float]:
    """
    Extracts the Pauli error rates from the measured fidelity pairs. By solving `Mx = -log(f)/(2t)`.
    Where `x` is the rates, `f` the measured fidelity pairs, and `M` is a matrix based on the
    commutation between the measured fidelity pairs and the Pauli's used in the model and can be
    expressed as `M = M_1 + M_2` (where `M_1`/`M_2` correspond to the first and second part of the
    fidelity pairs respectively and are constructed using the `matrix_m` function). This matrix
    needs to be column rank. This matrix will not be column rank just from measuring fidelity pairs
    and the fidelity pairs will need to be extended by approximations or single layer fidelity
    estimations.

    Parameters
    ----------
    fidelity_pairs : dict[str, float]
        The measured fidelity pairs.
    split_fidelity_pairs : bool, optional
        Whether to split the fidelity pairs, by default False
    non_negative : bool, optional
        Flag to force extracted rates to be non-negative, by default True

    Returns
    -------
    dict[str, float]
        The Pauli error rates (in Hz).

    See Also
    --------
    `extract_pauli_rates_symmetry_condition`:
        For use of an approximation to ensure `M` is column rank.
    `noise_learning.fidelity_estimation.CZPauliFidelityMeasurements`:
        For measuring the fidelity pairs.
    `probabilities_from_rates`:
        To convert the error rates to the probability of an error occurring.
    """
    paulis_1 = [pair.split("-")[0] for pair in fidelity_pairs]
    paulis_2 = [pair.split("-")[1] for pair in fidelity_pairs]
    M1 = matrix_m(paulis_1, MODEL_PAULIS)
    M2 = matrix_m(paulis_2, MODEL_PAULIS)
    M = M1 + M2
    if not np.linalg.matrix_rank(M) == len(MODEL_PAULIS):
        raise ValueError(
            "Matrix M = (M1 + M2) is not column rank. More / different Pauli fidelity pairs are needed to extract the rates."
        )

    f = list(fidelity_pairs.values())

    # We need to solve: Mλ = -1/2 log(f)
    return _solve_nnls(M, f, non_negative=non_negative)


def extract_pauli_rates_symmetry_condition(
    fidelity_pairs: dict[str, dict[str, float]],
    non_negative: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Uses the "symmetry condition" approximation to extract the Pauli error rates.
    Splits the "IX-ZX", "IY-ZY", "XI-XZ", and "YI-YZ" fidelity pairs into single fidelities by using
    the approximation that `f_1 = f_2 = sqrt(f_1f_2)`. Then uses `_extract_pauli_rates_from_pairs` to
    extract the rates.

    Based on:
    van den Berg, E., Minev, Z.K., Kandala, A. et al (2023)
    https://doi.org/10.1038/s41567-023-02042-2

    Parameters
    ----------
    fidelity_pairs : dict[str, float]
        Measured fidelity pairs per qubit. Should be structured like:
        `fidelity_pairs["Qubit pair [0, 2]"] = {"IX-ZY": 0.88, "XI-YZ": 0.91, ...}`
    non_negative : bool, optional
        Flag to force extracted rates to be non-negative, by default True

    Returns
    -------
    dict[str, dict[str, float]]
        Pauli error rates (in Hz).

    See Also
    --------
    `_extract_pauli_rates_from_pairs`:
        For the extraction of the rates from the fidelity pairs.
    `noise_learning.fidelity_estimation.CZPauliFidelityMeasurements`:
        For measuring the fidelity pairs.
    `probabilities_from_rates`:
        To convert the error rates to the probability of an error occurring.
    """
    output: dict[str, dict[str, float]] = {}
    for qubit_pair, qubit_pair_data in fidelity_pairs.items():
        data_copy = qubit_pair_data.copy()  # such that it does not change everywhere
        for pair in ["IX-ZX", "IY-ZY", "XI-XZ", "YI-YZ"]:
            p1, p2 = pair.split("-")
            if p1 + "-II" not in data_copy:
                data_copy[p1 + "-II"] = data_copy[p2 + "-II"] = np.sqrt(data_copy[pair])

        # we already have XY-XY, and YX-YX measured, don't need to include XY-YX in the analysis
        data_copy.pop("XY-YX", None)
        output[qubit_pair] = _extract_pauli_rates_from_pairs(data_copy, non_negative=non_negative)
    return output


def _prob_from_rate(rate: float) -> float:
    """Formula to calculate probability of Pauli error."""
    return 0.5 - 0.5 * np.exp(-2 * CZ_DURATION * rate)


def probabilities_from_rates(rates: dict[str, float]) -> dict[str, float]:
    """
    Calculates the (independent) probabilities from the error rates.
    Using `p_i = (1 - exp(-2t*x_i))/2` where `x_i` are the error rates.

    IMPORTANT: these probabilities should be evaluated independently, as they are not limited to
    only a single error at a time. For the effective probabilities see
    `effective_probabilities_from_rates`.

    Parameters
    ----------
    rates : dict[str, float]
        Pauli error rates.

    Returns
    -------
    dict[str, float]
        The probabilities of a Pauli error occurring.

    See Also
    --------
    `_prob_from_rate`:
        For the actual implementation.
    """
    return {key: _prob_from_rate(rate) for key, rate in rates.items()}


def _fidelity_from_probs(pauli: str, probs: dict[str, float]) -> float:
    """Calculates Pauli fidelity from probabilities."""
    fidelity = 1
    for pauli_2, prob in probs.items():
        if not paulis_commute(pauli, pauli_2):
            fidelity *= 1 - 2 * prob
    return fidelity


def fidelities_from_probabilities(probs: dict[str, float]) -> dict[str, float]:
    """
    Estimates the Pauli fidelity from the (independent) error probabilities.
    Calculated as `f_i = product(1-2*p_a)` for all Paulis 'a' that do not commute with 'i'.

    Parameters
    ----------
    probs : dict[str, float]
        The Pauli error probabilities.

    Returns
    -------
    dict[str, float]
        Pauli fidelity estimates.

    See Also
    --------
    '_fidelity_from_probs`:
        For the actual implementation.
    """
    return {pauli: _fidelity_from_probs(pauli, probs) for pauli in probs}


def fidelities_from_rates(rates: dict[str, float]) -> dict[str, float]:
    """
    Calculates Pauli fidelity estimates from the error rates. This is done by first calculating the
    probabilities of an specific error occurring and then the fidelities.

    Parameters
    ----------
    rates : dict[str, float]
        The Pauli error rates.

    Returns
    -------
    dict[str, float]
        Pauli fidelity estimates

    See Also
    --------
    `probabilities_from_rates`:
        For calculating the probabilities from the error rates.
    `fidelities_from_probabilities`:
        For calculating the fidelities from the error probabilities.
    """
    return fidelities_from_probabilities(probabilities_from_rates(rates))


def effective_probabilities_from_rates(rates: dict[str, float]) -> dict[str, float]:
    """
    Calculates the effective probabilities for a specific Pauli error to occur.
    This is done by calculating the effective probability and Pauli error for each
    combination of errors given by the independent error probabilities.

    Parameters
    ----------
    rates : dict[str, float]
        The Pauli error rates.

    Returns
    -------
    dict[str, float]
        The effective error rates.
    """
    # these are independent probabilities (multiple errors can happen at once)
    probs = probabilities_from_rates(rates)

    # calculate the effective probabilities
    effective_probs = {pauli: 0.0 for pauli in probs}
    effective_probs["II"] = 0.0

    # bitstrings correspond to each possible combination of errors that can occur
    # calculate the effective probability and Pauli for all these errors
    bitstrings = ["".join(i) for i in itertools.product("01", repeat=len(probs))]
    for bitstring in bitstrings:
        # initial probability and Pauli.
        p = 1
        pauli = Pauli("II")

        for idx, (pauli_str, prob) in enumerate(probs.items()):
            if bitstring[idx] == "0":
                p *= 1 - prob
                continue
            pauli = pauli.compose(Pauli(pauli_str))
            p *= prob

        # consider resulting pauli upto global phase
        pauli_str = pauli.to_label().strip("-i")
        effective_probs[pauli_str] += p
    return effective_probs


def plot_rates(
    rates: dict[str, float],
    title: str = "Pauli Error Rates",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Makes a bar plot visualising the Pauli error rates.

    Parameters
    ----------
    rates : dict[str, float]
        The Pauli error rates
    title : str, optional
        The title used for the figure, by default "Pauli Error Rates"

    Returns
    -------
    The figure and axes
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    set_colour_cycle_10(ax)
    x_values = list(rates.keys())
    y_values = np.array(list(rates.values()))

    ax.bar(x_values, y_values / 1000)
    ax.set_title(title)
    ax.set_ylabel("Rate (1/ms)")
    ax.set_xlabel("Paulis")
    return fig, ax


def plot_model_fidelity_pairs(
    rates: dict[str, float],
    measured_fidelity_pairs: dict[str, float],
    measured_fidelity_pairs_uncertainty: dict[str, float] | None = None,
    title: str = "Fidelity Pairs",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Makes a bar plot of the measured fidelity pairs, and the fidelity pair estimates from the Pauli
    channel error rates.

    Parameters
    ----------
    rates : dict[str, float]
        The error rates of the Pauli channel, extracted from the measured fidelity pairs.
    measured_fidelity_pairs : dict[str, float]
        The measured fidelity pairs
    measured_fidelity_pairs_uncertainty : dict[str, float] | None, optional
        The uncertainty of the measured fidelity pairs, used for error bars. Defaults to None, which
        results in no error bars.
    title : str, optional
        The title used for the figure, by default "Fidelity Pairs"

    Returns
    -------
    The figure and axes
    """
    model_fidelities = fidelities_from_rates(rates)
    model_fidelity_pairs = {}
    for pair in measured_fidelity_pairs:
        [pauli_1, pauli_2] = pair.split("-")
        model_fidelity_pairs[pair] = model_fidelities[pauli_1] * model_fidelities[pauli_2]

    # make plot
    fig, ax = plt.subplots(figsize=(10, 5))
    set_colour_cycle_10(ax)
    x_labels = list(measured_fidelity_pairs.keys())
    data = {
        "Measured": 1 - np.array(list(measured_fidelity_pairs.values())),
        "Estimates from rates": 1 - np.array(list(model_fidelity_pairs.values())),
    }
    yerr = {}
    if measured_fidelity_pairs_uncertainty is not None:
        yerr["Measured"] = list(measured_fidelity_pairs_uncertainty.values())

    plot_multi_bar(ax, x_labels, data, yerr=yerr)
    ax.set_title(title)
    ax.set_ylabel("$1-f_1f_2$")
    ax.set_xlabel("Fidelity pairs")
    ax.legend(ncols=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    return fig, ax
