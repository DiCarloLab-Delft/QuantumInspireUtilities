"""
Noise amplification experiment, and analysis classes.

Authors: Jan Hemink
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt
from qi_utilities.utility_functions.raw_data_processing import (
    observable_expectation_values_Z_basis,
    obtain_binary_list,
)
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit_quantuminspire import cqasm
from tqdm.autonotebook import tqdm

from qi_utilities.error_mitigation.noise_amplification.helper_functions import (
    exponential,
    linear,
    perform_ZNE,
    twirl_and_amplify,
)
from qi_utilities.error_mitigation.toolkit.base_classes import (
    BaseClassExperiment,
    BaseClassNoiseAmplificationData,
)
from qi_utilities.error_mitigation.toolkit.helper_functions import (
    _convert_to_z_basis,
    align_cz_gates,
    stitch_circuits,
)

if TYPE_CHECKING:
    import matplotlib.figure
    from qiskit_quantuminspire.qi_backend import QIBackend


class NoiseAmplification(BaseClassExperiment):
    """
    Experiment to amplify the noise of CZ gates in the given circuit by performing
    probabilistic error amplification (PEA).

    CZ gates in the circuit are Pauli twirled and additional Paulis are probabilistically sampled
    from the Pauli channel noise model and insert them before the CZ gates. Results are averaged
    over different randomizations to obtain results with an effective noise gain. Multiple different
    randomizations are combined into a circuit before it is executed on the backend to reduce
    execution time.

    Using results obtained at multiple noise gains, zero-noise extrapolation (ZNE) can be performed
    to obtain a noise mitigated estimate of the expectation values.

    Results and figures are saved to `{BASE_DIR}/{date}/{time}_{EXPERIMENT_NAME}`.

    Extends `BaseClassExperiment`.

    Experiment is based on
    This experiment is based on probabilistic error amplification as shown in:
    Kim, Y., Eddins, A., Anand, S. et al (2023)
    https://doi.org/10.1038/s41586-023-06096-3

    See Also
    --------
    `ZNE` : for performing ZNE on the noise amplified data.
    `qi_error_mitigation.noise_learning` : sub-module containing tools to learn the noise of CZ
        gates (and extract the Pauli error rates).
    """

    EXPERIMENT_NAME = "noise_amplification"

    def __init__(
        self,
        backend: QIBackend,
        transpiled_circuit: QuantumCircuit,
        qubit_list: list[int],
    ) -> None:
        """
        Parameters
        ----------
        backend : QIBackend
            The QI backend used to execute the circuits.
        transpiled_circuit : QuantumCircuit
            The quantum circuit to execute. Needs to be already transpiled for the selected backend.
        qubit_list : list[int]
            An ordered list of the qubits in the quantum circuit.
        """
        self.qc = transpiled_circuit
        self.nr_measurement_blocks = transpiled_circuit.num_clbits // len(qubit_list)
        if transpiled_circuit.num_clbits % len(qubit_list) != 0:
            raise ValueError("The number of classical bits of the circuit is not divisible by the number of qubits in the provided qubit_list")
        super().__init__(
            backend=backend,
            qubit_list=qubit_list,
        )

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pauli_error_rates: dict[str, dict[str, float]],
        noise_levels: list[float] = [1.0, 1.2, 1.6],
        num_shots: int = 64,
        num_random_instances: int = 2000,
        num_stitched_circuits: int = 40,
        ro_matrix_num_shots: int = 2**12,
        align_cz_gates: bool = False,
        timeout: int = 60 * 60,
        max_circuits_per_job: int = 1,
    ) -> tuple[dict, dict]:
        """
        Starts the noise amplification experiment.

        Parameters
        ----------
        pauli_error_rates : dict[str, dict[str, float]]
            The Pauli error rates used for PEA.
        noise_levels : list[float], optional
            The noise gains to use, by default [1., 1.2, 1.6].
        num_shots : int, optional
            The number of shots to used, by default 64.
        num_random_instances : int, optional
            The number of randomizations to use for PEA and Pauli twirling, by default 2000.
        num_stitched_circuits : int, optional
            The number of random instances that get combined into a single circuit before executing
            on the backend. Executing larger circuits with more mid-circuit measurements can reduce
            execution time compared to multiple smaller circuits. This value can not be 'too' high,
            as that will result in the circuit becoming too large for the control hardware memory.
            Defaults to 100.
        ro_matrix_num_shots : int, optional
            The number of shots used to calibrate the readout assignment matrices, by default 2**12.
        align_cz_gates : bool, optional
            If set to True, will align CZ gates into two-qubit gate layers, such that CZ gates on
            different qubit pairs happen in parallel at the exact same time, by default False.
        timeout : int, optional
            The timeout time (in seconds) used for `job.result(timeout=timeout)`, by default 1 hour.
        max_circuits_per_job : int, optional
            The maximum number of circuits to execute per job on the QuantumInspire backend,
            by default 1.

        Returns
        -------
        tuple[dict, dict]
            Returns the raw measurement probabilities and the readout corrected probabilities
            (both averaged over the randomizations).
        """
        # check inputs
        for noise_level in noise_levels:
            if noise_level < 1:
                raise ValueError("Noise levels should be larger or equal to 1.")

        # assign parameters
        self.noise_levels = noise_levels
        self.error_rates = pauli_error_rates
        self.align_cz_gates = align_cz_gates
        self.EXPERIMENT_NAME = f"{self.qc.name}_noise_amplification_G={self.noise_levels}"

        # run experiment
        return super().run(
            num_shots=num_shots,
            num_random_instances=num_random_instances,
            num_stitched_circuits=num_stitched_circuits,
            ro_matrix_num_shots=ro_matrix_num_shots,
            timeout=timeout,
            max_circuits_per_job=max_circuits_per_job,
        )

    def _create_project_data(self) -> None:
        """Stores project data to project folder."""
        super()._create_project_data()
        self.project_data["Experiment metadata"] = {
            "Noise levels": self.noise_levels,
            "Qubits": self.qubit_list,
            "Number of random instances": self.num_random_instances,
            "Number of circuits stitched together": self.num_stitched_circuits,
            "Number of measurement blocks": self.nr_measurement_blocks,
            "Number of shots": self.num_shots,
            "Readout assignment matrix number of shots": self.ro_matrix_num_shots,
            "Pauli error rates used [1/s]": self.error_rates,
            "Job directories": "",
        }
        self.project_data["Processed data"] = {
            "Total number of averages (num_shots * num_random_instances)": self.num_random_instances * self.num_shots,
            "Raw probabilities": {
                f"G={noise_level}": "" for noise_level in self.noise_levels
            },
            "Readout corrected probabilities": {
                f"G={noise_level}": "" for noise_level in self.noise_levels
            },
        }
        # save the provided circuit
        self._store_circuit()

    def _store_circuit(self) -> None:
        """Save the provided circuit to the project folder."""
        qc_qasm3 = qasm3.dumps(self.qc)
        qc_cqasm_v3 = cqasm.dumps(self.qc)
        qasm3_path = self.project_dir / f"quantum_circuit_qasm3_program_{self.timestamp}.qasm"
        cqasm_v3_path = self.project_dir / f"quantum_circuit_cqasm_v3_program_{self.timestamp}.cq"
        with qasm3_path.open("w") as f:
            f.write(qc_qasm3)
        with cqasm_v3_path.open("w") as f:
            f.write(qc_cqasm_v3)

        # also save a figure
        qc_fig: matplotlib.figure.Figure = self.qc.draw("mpl", scale=1.3)  # pyright: ignore[reportAssignmentType]
        qc_fig.suptitle(f"\n{self.timestamp}\nTranspiled quantum circuit\nCircuit name: {self.qc.name}\n", x = 0.5, y = 0.99, fontsize=16)
        qc_fig.supxlabel(f"Circuit depth: {self.qc.depth()}", x=0.5, y=0.06, fontsize=18)
        circuit_fig_path = self.project_dir / f"quantum_circuit_{self.timestamp}.png"
        qc_fig.savefig(circuit_fig_path)

    def _prepare_circuits(self) -> None:
        """Generates all circuits for the noise amplification experiment."""
        circuits = {}
        self.qc.barrier() # apply a barrier at the end
        stitched_circuit = stitch_circuits([self.qc] * self.num_stitched_circuits)
        for noise_level in self.noise_levels:
            stitched_circuit.name = f"{self.qc.name}_G={noise_level}"
            qc_list = [
                twirl_and_amplify(
                    qc=stitched_circuit,
                    noise_amplification=noise_level,
                    error_rates=self.error_rates,
                )
                for _ in range(math.ceil(self.num_random_instances / self.num_stitched_circuits))
            ]
            # optionally align CZ gates into layers where they happen at the exact same time on different qubits
            if self.align_cz_gates:
                qc_list = [align_cz_gates(qc, parallel=True) for qc in qc_list]
            circuits[noise_level] = transpile(
                qc_list,
                self.backend,
                basis_gates=self.NATIVE_GATES,
                routing_method="none",
                layout_method="trivial",
            )
        self.circuits = circuits

    def _execute_circuits(self) -> tuple[dict, dict]:
        """Executes all circuits and analyses the raw data."""
        raw_probabilities = {}
        probabilities = {}
        for noise_level in tqdm(self.circuits, desc="Noise Levels",bar_format=self.PROGRESSBAR_FORMAT,position=0):
            # save the noise level to the JSON containing the data from each job
            job_metadata = {"Noise level": noise_level}
            counts, raw_probs, probs = self._execute_jobs(self.circuits[noise_level], f"G={noise_level}", job_metadata)

            # turn it into a 1D list
            raw_probs = list(itertools.chain.from_iterable(raw_probs))
            probs = list(itertools.chain.from_iterable(probs))

            # when the circuit includes mid circuit measurements, group the probabilities per measurement block
            raw_probs = [raw_probs[idx :: self.nr_measurement_blocks] for idx in range(self.nr_measurement_blocks)]
            probs = [probs[idx :: self.nr_measurement_blocks] for idx in range(self.nr_measurement_blocks)]

            raw_probabilities[f"G={noise_level}"] = raw_probs
            probabilities[f"G={noise_level}"] = probs

        # average over the randomizations per noise level
        averaged_probs = {
            noise_level: [
                {
                    bitstring: np.mean([probs[bitstring] for probs in probs_list])
                    for bitstring in obtain_binary_list(len(self.qubit_list))
                }
                for probs_list in probabilities[noise_level]
            ]
            for noise_level in probabilities
        }
        averaged_raw_probs = {
            noise_level: [
                {
                    bitstring: np.mean([probs[bitstring] for probs in probs_list])
                    for bitstring in obtain_binary_list(len(self.qubit_list))
                }
                for probs_list in raw_probabilities[noise_level]
            ]
            for noise_level in raw_probabilities
        }

        self.project_data["Processed data"]["Raw probabilities"] = averaged_raw_probs
        self.project_data["Processed data"]["Readout corrected probabilities"] = averaged_probs
        self._store_project_data()
        return averaged_raw_probs, averaged_probs


class ZNE(BaseClassNoiseAmplificationData):
    """
    Analysis that performs zero-noise extrapolation (ZNE) on expectation values obtained at multiple
    noise gains. Can perform either linear or exponential extrapolation.

    Extends `BaseClassNoiseAmplificationData`.
    """

    def run(
        self,
        observables: list[list[str]],
        method: Literal["lin", "exp", "auto"] = "auto",
        plot_figures: bool = True,
    ) -> list[dict[str, float]]:
        """
        Load the data and perform analysis.

        Parameters
        ----------
        observables : list[str]
            The observables to use for ZNE. Specified per measurement block in the circuit.
            Eg. [[ZZ, IZ], [ZI]] would perform ZNE for the ZZ and IZ observables of the first
            measurement block and ZI of the second measurement block. If the circuit has no
            mid-circuit measurements then there is only one measurement block.
        method : Literal["lin", "exp", "auto"], optional
            Whether to use a linear or exponential fit. When set to `"auto"` will take whichever has
            the lowest sum of squared residuals, by default "auto".
        plot_figures : bool, optional
            Whether to create figures or not, by default True.

        Returns
        -------
        dict[str, float]
            The mitigated expectation values per measurement block of the circuit.
        """
        # load data
        self._load_data()

        # check input
        for observable_list in observables:
            for observable in observable_list:
                if len(observable) > len(self.qubits):
                    raise ValueError("Observables cannot be longer than the number of qubits.")
        if len(observables) != self.nr_measurement_blocks:
            raise ValueError(f"Not enough observables provided for the number of measurement blocks in the circuit ({self.nr_measurement_blocks}).")
        if method not in ["lin", "exp", "auto"]:
            raise ValueError("`method` must be one of ['lin', 'exp', 'auto'].")

        # compatibility older versions
        for noise_level in self.probabilities:
            if isinstance(self.probabilities[noise_level], dict):
                self.probabilities[noise_level] = [self.probabilities[noise_level]] # pyright: ignore[reportArgumentType]

        # assign parameters
        self.observables = observables
        self.method: Literal["lin", "exp", "auto"] = method

        # expand project data
        if "ZNE" not in self._project_data["Processed data"]:
            self._project_data["Processed data"]["ZNE"] = {
                "Mitigated expectation values": [{} for _ in range(self.nr_measurement_blocks)],
                "Expectation values": [{} for _ in range(self.nr_measurement_blocks)],
                "Fit results": [{} for _ in range(self.nr_measurement_blocks)],
                "Expectation values uncertainties": [{} for _ in range(self.nr_measurement_blocks)],
                "Mitigated values uncertainties": [{} for _ in range(self.nr_measurement_blocks)],
            }

        # run analysis
        output = self._perform_ZNE()
        self._store_project_data()
        if plot_figures:
            ZNEFigures(self.project_dir).run()
        return output

    def _perform_ZNE(self) -> list[dict[str, float]]:
        mitigated_values_list = []
        expectation_value_list = []
        fit_results_list = []
        uncertainties_list = []
        mitigated_uncertainty_list = []

        for measurement_block_idx, observable_list in enumerate(self.observables):
            mitigated_expectation_values = {}
            expectation_values_dict = {}
            fit_results_dict = {}
            uncertainties_dict = {}
            mitigated_value_uncertainty = {}
            for observable in observable_list:
                # calculate and save the expectation values from the probabilities
                expectation_value = {
                    noise_level: observable_expectation_values_Z_basis([probs_list[measurement_block_idx]], _convert_to_z_basis(observable))[0]
                    for noise_level, probs_list in self.probabilities.items()
                }
                # calculate standard error on the expectation values
                uncertainties = {
                    noise_level: np.sqrt((1-value**2)/(self.num_random_instances * self.num_shots))
                    for noise_level, value in expectation_value.items()
                }
                expectation_values_dict[observable] = expectation_value
                uncertainties_dict[observable] = uncertainties

                extrapolated, fit_results = perform_ZNE(
                    expectation_values=expectation_value,
                    method=self.method,
                    uncertainties=uncertainties,
                )

                if fit_results["Chosen fit"] == "Exponential":
                    mitigated_value_uncertainty[observable] = fit_results["Exponential"]["Fit uncertainty"][0]
                else:
                    mitigated_value_uncertainty[observable] = fit_results["Linear"]["Fit uncertainty"][1]

                mitigated_expectation_values[observable] = extrapolated  # pyright: ignore[reportPossiblyUnboundVariable]
                fit_results_dict[observable] = fit_results

            mitigated_values_list.append(mitigated_expectation_values)
            expectation_value_list.append(expectation_values_dict)
            fit_results_list.append(fit_results_dict)
            mitigated_uncertainty_list.append(mitigated_value_uncertainty)
            uncertainties_list.append(uncertainties_dict)

        for idx in range(self.nr_measurement_blocks):
            self._project_data["Processed data"]["ZNE"]["Mitigated expectation values"][idx].update(mitigated_values_list[idx])
            self._project_data["Processed data"]["ZNE"]["Expectation values"][idx].update(expectation_value_list[idx])
            self._project_data["Processed data"]["ZNE"]["Fit results"][idx].update(fit_results_list[idx])
            self._project_data["Processed data"]["ZNE"]["Expectation values uncertainties"][idx].update(uncertainties_list[idx])
            self._project_data["Processed data"]["ZNE"]["Mitigated values uncertainties"][idx].update(mitigated_uncertainty_list[idx])
        return mitigated_values_list


class ZNEFigures(BaseClassNoiseAmplificationData):
    """
    Creates figures form the noise amplification and ZNE data.

    Extends `BaseClassNoiseAmplificationData`.
    """

    def run(self) -> None:
        """Attempts to load the data and make the figures."""
        self._load_data()
        self._plot_figures()

    def _load_data(self) -> None:
        """Loads data from project folder."""
        super()._load_data()
        ZNE_data = self._project_data["Processed data"].get("ZNE", self._project_data["Processed data"]) # compatibility with older versions
        self.mitigated_values: list[dict[str, float]] = ZNE_data["Mitigated expectation values"]
        self.expectation_values: list[dict[str, dict[str, float]]] = ZNE_data["Expectation values"]
        self.fit_results: list[dict[str, dict]] = ZNE_data["Fit results"]

        # compatibility older versions
        for noise_level, data in self.probabilities.items():
            if isinstance(data, dict):
                self.probabilities[noise_level] = [data] # pyright: ignore[reportArgumentType]
        if isinstance(self.mitigated_values, dict):
            self.mitigated_values = [self.mitigated_values]
        if isinstance(self.fit_results, dict):
            self.fit_results = [self.fit_results]

        self.observables: list[list[str]] = [list(expectation_values.keys()) for expectation_values in self.expectation_values]
        self.expectation_uncertainties = ZNE_data.get("Expectation values uncertainties", None)
        self.mitigated_uncertainties = ZNE_data.get("Mitigated values uncertainties", None)


    def _plot_figures(self) -> None:
        """Makes the different figures."""
        self._single_figs()
        self.overview_fig()

    def _single_figs(self) -> None:
        """Makes a figure showing the expectation values and fits for each observable individually."""
        for measurement_block in range(self.nr_measurement_blocks):
            for observable in self.observables[measurement_block]:
                fig_folder = self.project_dir / "ZNE_figures"
                if self.nr_measurement_blocks > 1:
                    fig_folder = fig_folder / f"measurement_block_{measurement_block}"
                fig_folder.mkdir(parents=True, exist_ok=True)
                # single figures
                fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", dpi=300)
                xdata = self.noise_levels
                x_lin_values = np.linspace(0, max(self.noise_levels) * 1.1, 100)
                ydata = [self.expectation_values[measurement_block][observable][f"G={noise_level}"] for noise_level in xdata]

                ax.scatter(xdata, ydata, marker="x", label=f"Measured {observable} expectation values",zorder=10)
                if self.expectation_uncertainties is not None:
                    yerr = [self.expectation_uncertainties[measurement_block][observable][f"G={noise_level}"] for noise_level in xdata]
                    ax.errorbar(xdata, ydata,yerr=yerr,capsize=3, fmt='none')


                # plot both linear and exponential fits
                lin_fit = self.fit_results[measurement_block][observable]["Linear"]
                exp_fit = self.fit_results[measurement_block][observable]["Exponential"]
                mitigated = self.mitigated_values[measurement_block][observable]

                if lin_fit != "":
                    ax.plot(
                        x_lin_values,
                        linear(x_lin_values, *lin_fit["Fit parameters"]),
                        label="Linear fit",
                        color="tab:orange",
                        alpha=0.75,
                    )
                if exp_fit != "":
                    ax.plot(
                        x_lin_values,
                        exponential(x_lin_values, *exp_fit["Fit parameters"]),
                        label="Exponential fit",
                        color="tab:green",
                        alpha=0.75,
                    )

                ax.scatter(
                    0,
                    mitigated,
                    label=f"Mitigated $\\langle {observable} \\rangle = {mitigated:.2f}$",
                    marker="o",
                    color="tab:red",
                    zorder=10,
                )
                if self.mitigated_uncertainties is not None:
                    ax.errorbar(
                        0,
                        mitigated,
                        yerr = self.mitigated_uncertainties[measurement_block][observable],
                        color="tab:red",
                        zorder=10,
                    )

                ax.set_xlabel("Noise gain (G)")
                ax.set_ylabel("Expectation value")
                ax.set_title(f"{observable}-observable ZNE\nCircuit randomization N = {self.num_random_instances}, num_shots = {self.num_shots}\n{self.timestamp}")
                ax.set_xlim(left=-0.1)
                ax.get_xaxis().set_ticks([0, *self.noise_levels])
                ax.legend()
                fig.savefig(fig_folder / f"{observable}-observable_ZNE_{self.timestamp}.png",dpi=300)
                plt.close(fig)

    def overview_fig(self) -> None:
        """Makes an overview figure showing the expectation values and (chosen) fit for each observable."""
        for measurement_block in range(self.nr_measurement_blocks):
            fig_folder = self.project_dir / "ZNE_figures"
            if self.nr_measurement_blocks > 1:
                fig_folder = fig_folder / f"measurement_block_{measurement_block}"
            fig_folder.mkdir(parents=True, exist_ok=True)
            # make figure
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", dpi=300)
            xdata = self.noise_levels
            x_lin_values = np.linspace(0, max(self.noise_levels) * 1.1, 100)
            funcs = {"Linear": linear, "Exponential": exponential}
            for idx, observable in enumerate(self.observables[measurement_block]):
                ydata = [self.expectation_values[measurement_block][observable][f"G={noise_level}"] for noise_level in xdata]
                ax.scatter(
                    xdata,
                    ydata,
                    label=f"Measured $\\langle {observable} \\rangle$",
                    marker="x",
                    color=f"C{idx}",
                    alpha=0.75,
                )

                chosen = self.fit_results[measurement_block][observable]["Chosen fit"]
                fit_params = self.fit_results[measurement_block][observable][chosen]["Fit parameters"]
                ax.plot(
                    x_lin_values,
                    funcs[chosen](x_lin_values, *fit_params),
                    color=f"C{idx}",
                    ls="--",
                    alpha=0.5,
                )
                ax.scatter(
                    0,
                    self.mitigated_values[measurement_block][observable],
                    color=f"C{idx}",
                    marker="o",
                    alpha=0.75,
                    label=f"Mitigated $\\langle {observable} \\rangle = {self.mitigated_values[measurement_block][observable]:.2f}$"
                )
                if self.mitigated_uncertainties is not None:
                    ax.errorbar(
                        0,
                        self.mitigated_values[measurement_block][observable],
                        yerr = self.mitigated_uncertainties[measurement_block][observable],
                        color=f"C{idx}",
                        alpha=0.75,
                        )

            ax.set_ylabel("Expectation value")
            ax.set_xlabel("Noise gain (G)")
            ax.set_title(f"All observables ZNE\nCircuit randomization N = {self.num_random_instances}, num_shots = {self.num_shots}\n{self.timestamp}")
            ax.set_xlim(left=-0.1)
            ax.get_xaxis().set_ticks([0, *self.noise_levels])
            ax.legend()
            fig.savefig(fig_folder / f"overview_ZNE_{self.timestamp}.png", dpi=300)
            plt.close(fig)
