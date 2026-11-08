"""
Experiment, analysis, and figure classes for the noise learning of a CZ gate.

Authors: Jan Hemink
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from qi_utilities.utility_functions.raw_data_processing import (
    observable_expectation_values_Z_basis,
)
from qiskit import QuantumCircuit, transpile
from tqdm.autonotebook import tqdm

from qi_utilities.error_mitigation.noise_learning.helper_functions import (
    _exponential,
    _measured_fidelities_per_basis,
    cz_pauli_fidelity_measurement_circuit,
    extract_pauli_rates_symmetry_condition,
    fit_fidelity_pairs,
    plot_model_fidelity_pairs,
    plot_rates,
)
from qi_utilities.error_mitigation.toolkit.base_classes import (
    BaseClassExperiment,
    BaseClassNoiseLearningData,
    BaseClassNoiseLearningFigures,
)
from qi_utilities.error_mitigation.toolkit.helper_functions import (
    _convert_to_z_basis,
    align_cz_gates,
    stitch_circuits,
)
from qi_utilities.error_mitigation.toolkit.pauli_twirling import pauli_twirl_cz
from qi_utilities.error_mitigation.toolkit.plotting import COLOURS20, set_colour_cycle_10

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from qiskit_quantuminspire.qi_backend import QIBackend


class CZPauliFidelityMeasurements(BaseClassExperiment):
    """
    Experiment to determine the Pauli fidelities of the noise channel acting on a CZ gate.

    Will execute circuits with a different number of CZ repetitions (specified by `k_values`) for
    different Pauli bases (`CZPauliFidelityMeasurements.BASIS`). Circuits are constructed by the
    `cz_pauli_fidelity_measurement_circuit` function. If the basis ends in 'r', no single qubit
    rotations are applied after the CZ gates to correct for the CZ gate changing the Pauli basis.

    Circuits with a different `k_value`, but the same basis are then combined into one circuit using
    mid-circuit measurements (stitching them together). This circuit is then Pauli twirled with
    `num_random_instances`. Multiple (`num_stitched_circuits`) of these random instances are then
    combined into a larger circuit (more mid-circuit measurements) before they are executed.

    Measurement of different observables for the bases will result in the fidelity pairs specified
    in `MEASURED_FIDELITIES_PER_BASIS`. Once the observables have been measured for the different
    values of k. An exponential (`_fit_function`) is then fit to the data points to extract the
    fidelity pair value.

    Results and figures are saved to `{BASE_DIR}/{date}/{time}_{EXPERIMENT_NAME}`

    Extends `BaseClassExperiment`.

    This experiment is based on the procedure for learning a two-qubit gate noise channel as seen in:
    van den Berg, E., Minev, Z.K., Kandala, A. et al (2023)
    https://doi.org/10.1038/s41567-023-02042-2

    See Also
    --------
    `cz_pauli_fidelity_measurement_circuit` : For the creation of the circuits.
    `pauli_twirling.pauli_twirl_cz` : For the Pauli twirling of the circuits.
    `_fit_function` : For the function being fit to the data.
    `noise_learning.pauli_rates.extract_pauli_rates` : To extract the error rates from the fitted fidelity pairs.
    """

    EXPERIMENT_NAME = "CZ_Pauli_fidelities"

    # controls in which basis measurements will be done, appending 'r' to the basis will disable the single qubit correction rotations.
    BASIS = ("ZZ", "XX", "YY", "XY", "YX", "XYr", "YXr")

    def __init__(
        self,
        backend: QIBackend,
        qubit_pairs: list[list[int]],
        k_values: list[int] = [0, 2, 4, 8, 12, 24, 48],
    ) -> None:
        """
        Parameters
        ----------
        backend : QIBackend
            The `QIBackend` on which to perform this experiment.
        qubit_pairs : list[list[int]]
            A list containing the qubit pairs for which to learn the noise. Qubits cannot be
            included more than once.
        k_values : list[int], optional
            A list containing the different number of CZ repetitions to use for the measurement circuits.
            All values of `k` should be divisible by `2`. Defaults to `[0,2,4,8,12,48]`.
        """
        qubits_used = []
        for qubit_pair in qubit_pairs:
            if len(qubit_pair) != 2:
                raise ValueError("Please make sure that the `qubit_pairs` contain only two qubit indices each.")
            q1, q2 = qubit_pair
            if q1 in qubits_used or q2 in qubits_used:
                raise ValueError("Please include each qubit only once in the qubit pairs.")
            qubits_used.extend(qubit_pair)
        for k in k_values:
            if k % 2 != 0:
                raise ValueError("Please make sure that all `k_values` are divisible by 2.")

        self.k_values = k_values
        self.qubit_pairs = qubit_pairs
        self.EXPERIMENT_NAME = self.EXPERIMENT_NAME + f"_qubits{qubit_pairs}"  # change this later

        super().__init__(
            backend=backend,
            qubit_list=qubits_used,
        )

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        num_shots: int,
        num_random_instances: int = 100,
        num_stitched_circuits: int = 10,
        parallel: bool = True,
        ro_matrix_num_shots: int = 4096,
        timeout: int = 60 * 60,
        max_circuits_per_job: int = 1,
        save_data_to_disk: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Starts executing the experiment.

        Parameters
        ----------
        num_shots : int,
            The number of shots used to execute the circuits.
        num_random_instances : int, optional
            The number of random instances (per basis and k_value) to use for the Pauli twirling.
            Defaults to 256.
        num_stitched_circuits : int, optional
            The number of random instances that get combined into a single circuit before executing on
            the backend. Executing larger circuits with more mid-circuit measurements can reduce
            execution time compared to multiple smaller circuits. This value can not be 'too' high, as
            that will result in the circuit becoming too large for the control hardware memory.
            Defaults to 10, which works for the default `k_values`, different `k_values` can result in
            a larger (or smaller) circuit.
        parallel : bool, optional
            When True and measuring multiple qubit pairs, will perform CZ gates on different pairs
            at the same time. When False, will perform CZ gates sequentially.
        ro_matrix_num_shots : int, optional
            The number of shots used to calibrate the readout assignment matrices, by default 4096.
        timeout : int, optional
            The timeout time (in seconds) used for `job.result(timeout=timeout)`, by default 1 hour
        max_circuits_per_job: int, optional
            The maximum number of circuits to execute per job on the QuantumInspire backend.
        save_data_to_disk : bool, optional
            When True, will save data obtained by every circuit.
            When False, will only save final results. Defaults to True.

        Returns
        -------
        dict[str, dict[str, float]]
            The extracted Pauli error rates per qubit pair.
        """
        self.parallel = parallel
        return super().run(
            num_shots=num_shots,
            num_random_instances=num_random_instances,
            num_stitched_circuits=num_stitched_circuits,
            ro_matrix_num_shots=ro_matrix_num_shots,
            timeout=timeout,
            max_circuits_per_job=max_circuits_per_job,
            save_data_to_disk=save_data_to_disk,
        )

    def _create_project_data(self) -> None:
        super()._create_project_data()
        self.project_data["Experiment metadata"] = {
            "Pauli bases": self.BASIS,
            "Qubit pairs": self.qubit_pairs,
            "Number of random instances (per basis/k_value)": self.num_random_instances,
            "Number of circuits stitched together": self.num_stitched_circuits,
            "Number of shots": self.num_shots,
            "Readout assignment matrix number of shots": self.ro_matrix_num_shots,
            "k values": self.k_values,
            "Parallel CZ gates": self.parallel,
            "Job directories": "",
        }
        self.project_data["Processed data"] = {
            "Expectation values": "",
            "Fitted fidelity pairs": "",
            "Pauli error rates [1/s]": "",
            "Pauli error rates (negatives allowed) [1/s]": "",
            "Fit results": "",
        }

    def _prepare_circuits(self) -> None:
        """Generates all circuits for measuring the fidelities of the Pauli channel."""
        self.circuits: dict[str, list[QuantumCircuit]] = {}
        for basis in self.BASIS:
            name = f"{basis}-basis_Pauli_fidelities"

            # don't do single qubit rotations if 'r' in basis
            rotations = "r" in basis
            # combine circuits for all k values into one
            sub_qcs = [
                cz_pauli_fidelity_measurement_circuit(basis.strip("r") * len(self.qubit_pairs), k, rotations, name)
                for k in self.k_values
            ]
            qc = stitch_circuits(sub_qcs)
            # combine multiple random instances into one circuit
            stitched_qc = stitch_circuits([qc] * self.num_stitched_circuits)
            circuits = pauli_twirl_cz(stitched_qc, self.num_random_instances // self.num_stitched_circuits)
            if len(self.qubit_pairs) > 1:
                circuits = [align_cz_gates(circ, self.parallel) for circ in circuits]
            self.circuits[basis] = transpile(
                circuits=circuits,
                backend=self.backend,
                basis_gates=self.NATIVE_GATES,
                layout_method="trivial",
                routing_method="none",
                initial_layout=self.qubit_list,
            )

    def _execute_circuits(self) -> None:
        """Execute all circuits and analyse raw data."""
        expectation_values = {f"Qubit pair {pair}": {} for pair in self.qubit_pairs}
        # run circuit for each basis and k value
        for basis in tqdm(self.circuits, desc="Basis", position=0, bar_format=self.PROGRESSBAR_FORMAT):
            probabilities = {f"Qubit pair {pair}": {k: [] for k in self.k_values} for pair in self.qubit_pairs}
            job_metadata = {"Basis": basis, "k values": self.k_values}
            counts, raw_probs, probs = self._execute_jobs(self.circuits[basis], f"{basis}-basis", job_metadata)

            # turn it into a 1D list
            probs = list(itertools.chain.from_iterable(probs))

            # if multiple qubit pairs, split up probabilities per pair
            probs_per_qubit = {f"Qubit pair {pair}": [] for pair in self.qubit_pairs}
            if len(self.qubit_pairs) > 1:
                for idx, pair in enumerate(self.qubit_pairs):
                    for prob in probs:
                        temp = {"00": 0, "01": 0, "10": 0, "11": 0}
                        for bitstring, p in prob.items():
                            n = len(bitstring)
                            key = bitstring[n - 2 * (idx + 1) : n - 2 * idx]
                            temp[key] += p
                        probs_per_qubit[f"Qubit pair {pair}"].append(temp)
            else:
                probs_per_qubit = {f"Qubit pair {self.qubit_pairs[0]}": probs}

            # split probabilities up per k value and calculate observables
            for pair in probs_per_qubit:
                for idx, k in enumerate(self.k_values):
                    probabilities[pair][k].extend(probs_per_qubit[pair][idx :: len(self.k_values)])
                expectation_values[pair][f"{basis} basis"] = self._calculate_observables_from_probs(probabilities[pair], basis=basis)

        self.project_data["Processed data"]["Expectation values"] = expectation_values
        self._store_project_data()

    def _calculate_observables_from_probs(
        self,
        probabilities: dict[int, list[dict]],
        basis: str,
    ) -> dict:
        """Calculate the expectation values from the probabilities."""
        analysed_data = {}
        rotations = "r" not in basis
        # measuring the IZ, ZI, and ZZ observable gives different products of fidelities.
        for observable, fidelity_name in _measured_fidelities_per_basis(basis.strip("r"), rotations).items():
            analysed_data[fidelity_name] = []
            for k, probs_list in probabilities.items():
                # take the mean, as we did pauli twirling
                observ_list = observable_expectation_values_Z_basis(probs_list, _convert_to_z_basis(observable))
                fidelity = np.mean(observ_list)
                analysed_data[fidelity_name].append(fidelity)
        return analysed_data

    def _run_analysis(self) -> dict[str, dict[str, float]]:
        """Run further data analysis."""
        FidelityPairAnalysis(self.project_dir).run()
        self.rates = PauliRatesAnalysis(self.project_dir).run()
        return self.rates


class FidelityPairAnalysis(BaseClassNoiseLearningData):
    """
    Analysis that extracts the Pauli Fidelity Pairs from the measured data, by fitting an
    exponential function.

    Extends `BaseClassProjectData`.

    Parameters
    ----------
    project_dir: Path
        Path object of the project folder.

    See Also
    --------
    `FidelityPairFigures` :
        For the class making figures of the measured data and exponential fits.
    `noise_learning.pauli_rates.PauliRatesAnalysis` :
        For the analysis class that extracts the error rates from the fidelity pairs.
    """

    def run(self, plot_figures: bool = True) -> dict[str, dict[str, float]]:
        """
        Attempts to load the measured data from the project folder, then fits an exponential
        to the data points to obtain the fidelity pairs and save them (along with the fitting data)
        to the project folder. If `plot_figures` is True, will create a `FidelityPairFigures`
        instance and run it, creating figures of the measured data and exponential fits. If
        `extract_rates` is True, will create a `PauliRatesAnalysis` instance and run it,
        extracting the error rates from the saved fidelity pairs.

        Parameters
        ----------
        plot_figures : bool, optional
            Flag for making figures of the measured data and exponential fits, by default True

        Returns
        -------
        dict[str, dict[str, float]]
            The fitted fidelity pairs per qubit pair.

        See Also
        --------
        `FidelityPairFigures` :
            For the class making figures of the measured data and exponential fits.
        `noise_learning.pauli_rates.PauliRatesAnalysis` :
            For the analysis class that extracts the error rates from the fidelity pairs.
        """
        self._load_data()
        output = self._fit_fidelities()
        self._store_project_data()
        if plot_figures:
            FidelityPairFigures(self.project_dir).run(only_overview_figure=False)
        return output

    def _fit_fidelities(self) -> dict[str, dict[str, float]]:
        """Fit exponential to extract the fidelity pairs."""
        fitted_fidelities, fit_results = fit_fidelity_pairs(self.expectation_values, self.k_values)
        self._project_data["Processed data"]["Fit results"] = fit_results
        self._project_data["Processed data"]["Fitted fidelity pairs"] = fitted_fidelities
        self.fidelity_pairs = fitted_fidelities
        return fitted_fidelities


class FidelityPairFigures(BaseClassNoiseLearningFigures):
    """
    Figure plotting class that creates figures from measured data and exponential fits.

    Extends `BaseClassFigures`.

    Parameters
    ----------
    project_dir: Path
        Path object of the project folder.
    """

    def run(
        self,
        only_overview_figure: bool = True,
        fidelity_pairs_to_plot: Sequence[str] | Literal["all"] = "all",
        ylim: tuple[float | int, float | int] = (-0.1, 1.1),
    ) -> None:
        """
        Attempts to load the data and make the figures. If `only_overview_figure` is true, will only
        make the overview figure, else will also make fits for
        the individual Pauli bases used for measurements.

        Parameters
        ----------
        only_overview_figure : bool, optional
            Flag to only plot the overview figure, if False will also make figures for the
            individual Pauli bases used in measurements, by default True
        fidelity_pairs_to_plot : Sequence[str] | Literal["all"], optional
            Which fidelity pair fits (and corresponding data points) to include in the
            overview figure. Defaults to "all", resulting in all measured fidelity pairs to be
            included.
        ylim : tuple[float | int, float | int], optional
            The limits of the y-axis used for the overview figure, is passed to `ax.set_ylim(ylim)`.
            Defaults to (-0.1, 1.1)
        """
        self.only_overview_figure = only_overview_figure
        self.fidelity_pairs_to_plot: Sequence[str] | Literal["all"] = fidelity_pairs_to_plot
        self.ylim = ylim
        super().run()

    def _load_data(self) -> None:
        super()._load_data()
        if self.fidelity_pairs is None:
            raise ValueError(f"Fitted fidelity pairs were not found in {self.project_dir}")
        self.fit_results: dict[str, dict] = self._project_data["Processed data"]["Fit results"]

    def _make_dir(self, qubit_pair: list[int]) -> None:
        qubit_pair_dir = self.project_dir / f"Qubit_pair_{qubit_pair}"
        qubit_pair_dir.mkdir(exist_ok=True)

    def _plot_figures(self) -> None:
        """Creates and saves figures."""
        # k_values used to plot fitted exponential
        k_lin_values = np.linspace(0, max(self.k_values) + 1, max(self.k_values) * 2)

        for qubit_pair in self.qubit_pairs:
            self._make_dir(qubit_pair)
            # individual figures
            if not self.only_overview_figure:
                self._individual_figures(k_lin_values, qubit_pair)
            # overview figure
            self._overview_figure(k_lin_values, qubit_pair)

    def _overview_figure(self, k_lin_values: NDArray, qubit_pair: list[int]) -> None:
        q1, q2 = qubit_pair
        dict_key = f"Qubit pair {qubit_pair}"

        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        all_fidelity_pairs = list(self.fidelity_pairs[dict_key].keys())  # pyright: ignore[reportOptionalSubscript]
        if self.fidelity_pairs_to_plot == "all":
            plot = all_fidelity_pairs
        else:
            plot = self.fidelity_pairs_to_plot

        for fidelity_pair in plot:
            data_points = self.fit_results[dict_key]["Combined expectation values"][fidelity_pair]
            fit_params = self.fit_results[dict_key]["Fit parameters"][fidelity_pair]
            std_error = self.fit_results[dict_key]["Fit uncertainty"][fidelity_pair]

            # ensure each fidelity pair keeps same colour, even when not plotting all of them
            colour = COLOURS20[all_fidelity_pairs.index(fidelity_pair) % 20]
            ax.scatter(self.k_values, data_points, zorder=10, color=colour)
            ax.plot(
                k_lin_values,
                _exponential(k_lin_values, *fit_params.values()),
                label=f"{fidelity_pair} fit, $f_1f_2$ = {fit_params['f1f2']:.4f} ± {std_error['f1f2']:.4f}",
                color=colour,
            )
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Fit: $a*(f_1f_2)^{k/2}+b$")
        ax.set_ylim(self.ylim)
        ax.set_xlabel("CZ repetitions, k")
        ax.set_ylabel("Expectation value")
        ax.set_title(self._title.format(name="CZ Pauli fidelity pairs", q1=q1, q2=q2))

        # check if overview figure already exist, we don't want to overwrite it.
        fig_path = self.project_dir / f"Qubit_pair_{qubit_pair}" / f"fidelity_pairs_q{q1}q{q2}_{self.timestamp}.png"
        if fig_path.exists():
            i = 1
            while fig_path.exists():
                fig_path = self.project_dir / f"Qubit_pair_{qubit_pair}" / f"fidelity_pairs_q{q1}q{q2}_{self.timestamp}({i}).png"
                i += 1

        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def _individual_figures(self, k_lin_values: NDArray, qubit_pair: list[int]) -> None:
        q1, q2 = qubit_pair
        for basis in self.basis:
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            set_colour_cycle_10(ax)
            text = "Fit: $a*(f_1f_2)^{k/2}+b$"
            rotations = "r" not in basis

            dict_key = f"Qubit pair {qubit_pair}"
            for fidelity_pair in _measured_fidelities_per_basis(basis.strip("r"), rotations).values():
                data_points = self.fit_results[dict_key]["Combined expectation values"][fidelity_pair]
                fit_params = self.fit_results[dict_key]["Fit parameters"][fidelity_pair]
                std_error = self.fit_results[dict_key]["Fit uncertainty"][fidelity_pair]
                SSR = self.fit_results[dict_key]["SSR"][fidelity_pair]
                R2 = self.fit_results[dict_key]["R2"][fidelity_pair]
                ax.scatter(self.k_values, data_points, zorder=10)
                ax.plot(
                    k_lin_values,
                    _exponential(k_lin_values, *fit_params.values()),
                    label=f"{fidelity_pair} fit, $f_1f_2$ = {fit_params['f1f2']:.4f} ± {std_error['f1f2']:.4f}",
                )
                text += f"\n\n{fidelity_pair}:\n$SSR = {SSR:.2e}$, $R^2 = {R2:.4g}$ \n$a = {fit_params['a']:.5g} ± {std_error['a']:.5g}$"
                text += f"\n$f_1f_2 = {fit_params['f1f2']:.5g} ± {std_error['f1f2']:.5g}$\n $b = {fit_params['b']:.5g} ± {std_error['b']:.5g}$"
            ax.set_ylim(top=1.1, bottom=-0.1)
            ax.legend()
            ax.set_xlabel("CZ repetitions, k")
            ax.set_ylabel("Expectation value")
            ax.set_title(self._title.format(name=f"CZ Pauli fidelity pairs {basis}-basis", q1=q1, q2=q2))
            # text box containing fitting information
            fig.text(1.01, 0.94, text, ha="left", va="top", bbox={"facecolor": "white"})

            dir_path = self.project_dir / f"Qubit_pair_{qubit_pair}" / f"individual_basis_plots_q{q1}q{q2}"
            if not dir_path.exists():
                dir_path.mkdir()
            fig.savefig(dir_path  / f"{basis}-basis_fidelity_pairs_q{q1}q{q2}_{self.timestamp}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


class PauliRatesAnalysis(BaseClassNoiseLearningData):
    """
    Analysis that extracts the Pauli error rates from the fitted fidelity pairs.

    Extends `BaseClass`.

    Parameters
    ----------
    project_dir: Path
        Path object of the project folder.

    See Also
    --------
    `PauliRatesFigures` :
        For the class making figures of the Pauli rates.
    `noise_learning.fidelity_estimation.FidelityPairAnalysis` :
        For the analysis class that obtains the fidelity pairs from the measured data.
    """

    def run(
        self,
        plot_figures: bool = True,
        only_non_negative: bool = False,
    ) -> dict[str, dict[str, float]]:
        """
        Attempts to load the fitted fidelities from the project folder. The Pauli rates are extracted
        and saved to the same project folder. If `plot_figures` is True, will create a
        `PauliRatesFigures` instance and run it, creating figures from the saved rates.

        Parameters
        ----------
        plot_figures : bool, optional
            Flag for making the figures, by default True
        only_non_negative : bool, optional
            When True only extracts the rates while bounded to be non-negative. When False, will
            additionally also extract the rates without this bound and save them separately.
            Defaults to False.

        See Also
        --------
        `PauliRatesFigures` :
            For the class making figures of the Pauli rates.
        `noise_learning.fidelity_estimation.FidelityPairAnalysis` :
            For the analysis class that obtains the fidelity pairs from the measured data.
        """
        self.only_non_negative = only_non_negative
        self._load_data()

        self._extract_rates()
        self._store_project_data()
        if plot_figures:
            PauliRatesFigures(self.project_dir).run()
        return self.error_rates

    def _extract_rates(self) -> None:
        if self.fidelity_pairs is None:
            raise ValueError(f"Fitted fidelity pairs were not found in {self.project_dir}")

        self.error_rates = extract_pauli_rates_symmetry_condition(
            fidelity_pairs=self.fidelity_pairs,
            non_negative=True,
        )
        if not self.only_non_negative:
            self.negative_error_rates = extract_pauli_rates_symmetry_condition(
                fidelity_pairs=self.fidelity_pairs,
                non_negative=False,
            )
        self._project_data["Processed data"]["Pauli error rates [1/s]"] = self.error_rates
        self._project_data["Processed data"]["Pauli error rates (negatives allowed) [1/s]"] = self.negative_error_rates


class PauliRatesFigures(BaseClassNoiseLearningFigures):
    """
    Figure plotting class that creates figures from the extracted Pauli rates.

    Extends `BaseClassFigures`.

    Parameters
    ----------
    project_dir: Path
        Path object of the project folder.
    """

    def run(
        self,
        ylim_rates: float | None = None,
        ylim_fidelity: float | None = None,
    ) -> None:
        """
        Attempts to load the data and make the figures.

        Parameters
        ----------
        ylim_rates: float | None, optional
            The top limit of the y-axis for the Pauli error rates figure, by default None
        ylim_fidelity: float | None, optional
            The top limit of the y-axis for the figure comparing the measured fidelity pairs to the
            estimates from the extracted rates, by default None
        """
        self.ylim_rates = ylim_rates
        self.ylim_fidelity = ylim_fidelity
        self._load_data()
        self._plot_figures()

    def _load_data(self) -> None:
        super()._load_data()
        self.error_rates: dict[str, dict[str, float]] = self._project_data["Processed data"]["Pauli error rates [1/s]"]
        self.negative_error_rates: dict[str, dict[str, float]] | None
        self.negative_error_rates = self._project_data["Processed data"].get("Pauli error rates (negatives allowed) [1/s]", None)

    def _make_dir(self, qubit_pair: list[int]) -> None:
        qubit_pair_dir = self.project_dir / f"Qubit_pair_{qubit_pair}"
        qubit_pair_dir.mkdir(exist_ok=True)

    def _plot_figures(self) -> None:
        for qubit_pair in self.qubit_pairs:
            self._make_dir(qubit_pair)
            self._plot_rates(qubit_pair)
            self._plot_model_fidelity_pairs(qubit_pair)

    def _plot_rates(self, qubit_pair: list[int]) -> None:
        q1, q2 = qubit_pair
        dict_key = f"Qubit pair {qubit_pair}"
        # non-negative rates
        title = self._title.format(name="Pauli error rates", q1=q1, q2=q2)
        fig, ax = plot_rates(self.error_rates[dict_key], title)
        ax.set_ylim(top=self.ylim_rates)
        fig.savefig(self.project_dir / f"Qubit_pair_{qubit_pair}" / f"pauli_rates_q{q1}q{q2}_{self.timestamp}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # also make figure for 'negative' rates if they are saved in the project folder.
        if isinstance(self.negative_error_rates, dict):
            title = self._title.format(name="Pauli error rates (negatives allowed)", q1=q1, q2=q2)
            fig, ax = plot_rates(self.negative_error_rates[dict_key], title)
            ax.axhline(0,lw=1,color='k',alpha=0.5,ls='-')
            fig.savefig(self.project_dir / f"Qubit_pair_{qubit_pair}" / f"allow_negative_pauli_rates_q{q1}q{q2}_{self.timestamp}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _plot_model_fidelity_pairs(self, qubit_pair: list[int]) -> None:
        q1, q2 = qubit_pair
        dict_key = f"Qubit pair {qubit_pair}"

        title = self._title.format(name="Model fidelity pairs", q1=q1, q2=q2)
        fig, ax = plot_model_fidelity_pairs(self.error_rates[dict_key], self.fidelity_pairs[dict_key], title=title) # pyright: ignore[reportOptionalSubscript]
        ax.set_ylim(top=self.ylim_fidelity)
        fig.savefig(self.project_dir / f"Qubit_pair_{qubit_pair}" / f"model_fidelity_pairs_q{q1}q{q2}_{self.timestamp}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
