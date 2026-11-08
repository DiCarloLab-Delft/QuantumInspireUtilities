"""
Noise amplification utility functions.

Authors: Jan Hemink
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from qiskit.circuit import Barrier
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from scipy.optimize import curve_fit

from qi_utilities.error_mitigation.noise_learning.helper_functions import _prob_from_rate
from qi_utilities.error_mitigation.toolkit.pauli_twirling import CZ_PAULI_TWIRL

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from qiskit import QuantumCircuit
    from qiskit.circuit.gate import Gate


pauli_gates: dict[str, Gate] = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}


def _get_pauli_gates(pauli: str) -> list:
    return [pauli_gates[letter] for letter in reversed(pauli)]


def twirl_and_amplify(
    qc: QuantumCircuit,
    noise_amplification: float,
    error_rates: dict[str, dict[str, float]],
) -> QuantumCircuit:
    """
    Performs Pauli twirling and PEA of CZ gates in the circuit.
    Resulting circuit still needs to be transpile.
    Expectation values should be averaged over different randomizations.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit.
    noise_amplification : float
        The noise gain used for PEA.
    error_rates : dict[str, dict[str, float]]
        The Pauli error rates used for PEA.

    Returns
    -------
    QuantumCircuit
        The twirled and noise amplified circuit.
    """
    if noise_amplification < 1:
        raise ValueError("`noise_amplification` should be larger than 1 for probabilistic error amplification.")

    output_qc = qc.copy_empty_like()
    for instruction in qc:
        if instruction.name != "cz":  # if not a CZ-gate, just copy it to new circuit
            output_qc._data.append(instruction)
            continue

        qubits = instruction.qubits

        # Noise amplification (PEA)
        if noise_amplification != 1:
            rates: dict[str, float] = error_rates.get(f"Qubit pair {[qubits[0]._index, qubits[1]._index]}", None) # type: ignore
            if rates is None:
                rates = error_rates.get(f"Qubit pair {[qubits[1]._index, qubits[0]._index]}")
            for pauli, rate in rates.items():
                scaled_rate = rate * (noise_amplification - 1)
                prob = _prob_from_rate(scaled_rate)
                error = random.choices([True, False], weights=[prob, 1 - prob])[0]
                if not error:
                    continue
                for idx, gate in enumerate(_get_pauli_gates(pauli)):
                    output_qc.append(gate, (qubits[idx],))

        # Pauli twirling
        pauli_gates, phase = random.choice(CZ_PAULI_TWIRL)  # choose random pauli gates
        if phase != 0:
            output_qc.global_phase += phase
        # add gates to new circuit
        output_qc.append(pauli_gates[0], (qubits[0],))
        output_qc.append(pauli_gates[1], (qubits[1],))
        output_qc.append(Barrier(2), qubits)
        output_qc._data.append(instruction)
        output_qc.append(Barrier(2), qubits)
        output_qc.append(pauli_gates[2], (qubits[0],))
        output_qc.append(pauli_gates[3], (qubits[1],))

    return output_qc


def load_error_rates_from_disk(
    backend_name: str,
    start_date: str,
    start_time: str | None = None,
    end_date: str | None = None,
    end_time: str | None = None,
    search_expr: str = "*CZ_Pauli_fidelities*",
    base_dir: Path = Path.home() / "Documents" / "QuantumInspireProjects",
) -> dict[str, dict[str, float]]:
    """
    Loads Pauli error rates obtained by previous experiments from disk for a specific QI backend.
    Loops over all date folders in `base_dir`, looking for subfolders matching `search_expr`.
    Attempts to load error rates from these folders if they were obtained on the desired backend.
    Returns the most recent error rates for each qubit pair obtained between `start_date` and
    `end_date`.

    Parameters
    ----------
    backend_name : str
        The name of the QI backend for which the Pauli error rates should be loaded.
    start_date : str
        Only searches for experiments started on or after this date.
    start_time : str | None, optional
        Only searches for experiments started after this time on `start_date`, by default None.
    end_date : str | None, optional
        Only searches for experiments up to (and including) this date, by default None.
    end_time : str | None, optional
        Only searches for experiments up to (and including) this time on `end_date`, by default None.
    search_expr : str, optional
        The search expression used to locate the experiment folders that have previously measured
        the Pauli error rates, by default `"*CZ_Pauli_fidelities*"`.
    base_dir : Path, optional
        The base directory used to store QI experiment data, by default
        `Path.home()/"Documents"/"QuantumInspireProjects"`.

    Returns
    -------
    dict[str, dict[str, float]]
        The most recent error rates found on disk, measured within the specified timeframe,
        for each qubit pair of the desired QI backend
    """
    # convert to int, such that we can compare them
    begin_date = int(start_date)
    begin_time = 0 if start_time is None else int(start_time)
    stop_date = int(datetime.datetime.now().strftime("%Y%m%d")) if end_date is None else int(end_date)
    stop_time = 999999 if end_time is None else int(end_time)

    output = {}

    for date_folder in sorted(folder for folder in base_dir.iterdir() if folder.is_dir()):
        date = date_folder.parts[-1]
        if not date.isdigit():
            continue
        date = int(date)
        if date < begin_date:
            continue
        if date > stop_date:
            continue
        for experiment_folder in sorted(date_folder.glob(search_expr)):
            time_str = experiment_folder.parts[-1].split("_",1)[0] # because the time might start with a 0, which gets removed by int()
            time = int(time_str)
            if (date == begin_date and time < begin_time) or (date == stop_date and time > stop_time):
                continue
            project_data_file = experiment_folder / f"project_data_{date}_{time_str}.json"
            if not project_data_file.exists():
                print(f"Could not find {project_data_file}")
                continue
            with project_data_file.open("r") as f:
                project_data = json.load(f)
            if project_data["Backend info"]["Backend name"] != backend_name:
                continue
            rates = project_data["Processed data"]["Pauli error rates [1/s]"]
            if rates != "":
                output.update(rates)

    return output


def _linear_fit(xdata: list[float], ydata: list[float], yerr: list | NDArray | None = None) -> dict:
    # flip sign of initial guess if expectation value trends to negative value
    p0 = [1, -1] if ydata[0] < ydata[-1] else [-1, 1]

    return _perform_fit(func=linear, xdata=np.array(xdata), ydata=np.array(ydata), p0=p0, yerr=yerr)


def linear(G, a, b):
    return a * G + b


def _exponential_fit(xdata: list[float], ydata: list[float], yerr: list | NDArray | None = None) -> dict:
    # flip sign of initial guess if expectation value trends to negative value
    p0 = [-1, 1] if ydata[0] < ydata[-1] else [1, 1]

    return _perform_fit(func=exponential, xdata=np.array(xdata), ydata=np.array(ydata), p0=p0, yerr=yerr)


def exponential(G, a, b):
    return a * b**-G


def _perform_fit(
    func: Callable,
    xdata: NDArray,
    ydata: NDArray,
    p0: list,
    yerr: list | NDArray | None = None,
) -> dict:
    params, cov = curve_fit(
        f=func,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        maxfev=10000,
        sigma=yerr,
        absolute_sigma=True,
    )
    uncertainties = np.sqrt(np.diag(cov))
    sum_squared_residual = np.sum((ydata - func(xdata, *params)) ** 2)
    sum_squares_total = np.sum((ydata - ydata.mean()) ** 2)
    r_square = 1 - sum_squared_residual / sum_squares_total

    extrapolated = func(0, *params)

    return {
        "Extrapolated value": extrapolated,
        "Fit parameters": params,
        "Fit uncertainty": uncertainties,
        "SSR": sum_squared_residual,
        "SST": sum_squares_total,
        "R2": r_square,
    }


def perform_ZNE(
    expectation_values: dict[str, float],
    method: Literal["lin", "exp", "auto"] = "auto",
    uncertainties: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """
    Performs zero-noise extrapolation (ZNE) on noise amplified data.

    Parameters
    ----------
    expectation_values : dict[str, float]
        Dictionary containing the expectation values for the circuit at different noise gains.
        Keys should have the form of "G={noise_level}", for example:
        `{"G=1.0": 0.84, "G=1.2": 0.79, "G=1.6": 0.71}`
    method : Literal["lin", "exp", "auto"], optional
        Whether to use a linear or exponential fit. When set to `"auto"` will take whichever has
        the lowest sum of squared residuals, by default "auto".
    uncertainties : dict[str, float] | None, optional
        The uncertainties/errors of the expectation values. Will be used for the fitting and to
        obtain a more accurate estimate of the uncertainty of the mitigated expectation value.
        Should have the same dictionary structure as the `expectation_values`.

    Returns
    -------
    float, dict
        Returns the ZNE mitigated expectation value, along with a dictionary containing fit results.
    """
    # pre-load dict, such that there are default values and a nice order for in the JSON
    fit_results: dict = {
        "Chosen fit": "",
        "Extrapolated value": "",
        "Linear": "",
        "Exponential": "",
    }

    if method == "Exponential":
        method = "exp"
    elif method == "Linear":
        method = "lin"

    # perform ZNE
    xdata = [float(key.split("=")[-1]) for key in expectation_values]  # noise levels
    ydata = list(expectation_values.values())
    yerr = [uncertainties[key] for key in expectation_values] if uncertainties is not None else None

    if method != "exp":
        lin_fit_results = _linear_fit(xdata, ydata, yerr)
        fit_results["Linear"] = lin_fit_results
        extrapolated: float = lin_fit_results["Extrapolated value"]
        chosen_fit = "Linear"

    if method != "lin":
        exp_fit_results = _exponential_fit(xdata, ydata, yerr)
        fit_results["Exponential"] = exp_fit_results
        if method == "exp" or (exp_fit_results["SSR"] < lin_fit_results["SSR"]):  # pyright: ignore[reportPossiblyUnboundVariable]
            extrapolated: float = exp_fit_results["Extrapolated value"]
            chosen_fit = "Exponential"

    fit_results["Chosen fit"] = chosen_fit  # pyright: ignore[reportPossiblyUnboundVariable]
    fit_results["Extrapolated value"] = extrapolated  # pyright: ignore[reportPossiblyUnboundVariable]

    return extrapolated, fit_results  # pyright: ignore[reportPossiblyUnboundVariable]
