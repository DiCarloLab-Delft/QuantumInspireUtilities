"""
Base classes for classes that need to load data from a project folder.

Authors: Jan Hemink
"""

from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from qi_utilities.utility_functions.raw_data_processing import (
    get_multi_counts,
    get_multi_probs,
    get_raw_data,
)
from qiskit import QuantumCircuit, transpile
from tqdm.autonotebook import tqdm

from qi_utilities.error_mitigation.toolkit.helper_functions import (
    NumpyEncoder,
    SaveJobData,
    create_job_and_get_result,
    get_job_execution_time,
    get_path_from_timestamp,
    stitch_circuits,
)
from qi_utilities.error_mitigation.toolkit.readout_error_mitigation import (
    analyse_readout_matrix_circuit,
    apply_readout_assignment_correction,
    readout_matrix_calibration_circuit,
)

if TYPE_CHECKING:
    from qiskit.result import Result
    from qiskit_quantuminspire.qi_backend import QIBackend


class BaseClassProjectData:
    """
    General base class for classes that need to load data from a project folder.

    Parameters
    ----------
    project_dir : Path
        Path object of the project folder.
    """

    FILE_NAME = "project_data_????????_??????.json"

    def __init__(self, project_dir: Path) -> None:
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Could not find folder {project_dir}")
        self.project_dir = project_dir

    def _store_project_data(self) -> None:
        self._store_json(self._project_data, f"project_data_{self.timestamp}")

    def _store_json(self, data: dict, name: str) -> None:
        """Saves data to JSON file in project folder."""
        with Path.open(self.project_dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=3, cls=NumpyEncoder)

    def _load_data(self) -> None:
        """Load experiment data from project folder."""
        file = None
        for file in self.project_dir.glob(self.FILE_NAME):
            break
        if not file:
            raise FileNotFoundError(f"Could not find project data JSON file in {self.project_dir}")

        with file.open("r") as f:
            self._project_data: dict = json.load(f)
        self.experiment_name: str = self._project_data["Experiment name"]
        self.timestamp: str = self._project_data["Timestamp"]
        self.backend_name: str = self._project_data["Backend info"]["Backend name"]
        self.total_time: float = self._project_data["Total experiment duration [min]"]
        self.backend_time: float = self._project_data["Execution time backend [min]"]
        self.num_random_instances: int = self._project_data["Experiment metadata"].get(
            "Number of random instances (per basis/k_value)",
            self._project_data["Experiment metadata"].get("Number of random instances", None),
        )
        self.num_shots: int = self._project_data["Experiment metadata"]["Number of shots"]

    @classmethod
    def from_timestamp(
        cls,
        timestamp: str,
        base_dir: Path = Path.home() / "Documents" / "QuantumInspireProjects",
    ) -> Self:
        """
        Tries to get the `Path` to the project folder from the `date` and `time` of the experiment,
        then uses it to initialise an instance of this class.

        Parameters
        ----------
        timestamp : str
            The timestamp at which the experiment was performed, in string format: 'yyyymmdd_HHMMSS',
            e.g. '20260219_155500' (February 2nd 2026 15:55:00).
        base_dir : Path, optional
            The base directory used to store all project folders, by default Path.home()/"Documents"/"QuantumInspireProjects".

        Returns
        -------
        An instance of this class.
        """
        path = get_path_from_timestamp(timestamp, base_dir=base_dir)
        return cls(path)


class BaseClassNoiseLearningData(BaseClassProjectData):
    """
    General base class for classes that need to load noise learning data from a project folder.

    Parameters
    ----------
    project_dir : Path
        Path object of the project folder.
    """

    def _load_data(self) -> None:
        super()._load_data()

        self.basis: list[str] = self._project_data["Experiment metadata"]["Pauli bases"]
        self.qubit_pairs: list[list[int]] = self._project_data["Experiment metadata"]["Qubit pairs"]
        self.k_values: list[int] = self._project_data["Experiment metadata"]["k values"]

        self.expectation_values: dict[str, dict[str, dict[str, float]]]
        self.expectation_values = self._project_data["Processed data"]["Expectation values"]

        # load the fidelity pairs if they have been obtained from analysis
        self.fidelity_pairs: dict[str, dict[str, float]] | None
        self.fidelity_pairs = self._project_data["Processed data"].get("Fitted fidelity pairs", None)


class BaseClassNoiseAmplificationData(BaseClassProjectData):
    """
    General base class for classes that need to load noise amplification data from a project folder.

    Parameters
    ----------
    project_dir : Path
        Path object of the project folder.
    """

    def _load_data(self) -> None:
        super()._load_data()

        self.noise_levels: list[float] = self._project_data["Experiment metadata"]["Noise levels"]
        self.qubits: list[int] = self._project_data["Experiment metadata"]["Qubits"]
        self.probabilities: dict[str, list[dict[str, float]]] = self._project_data["Processed data"]["Readout corrected probabilities"]
        self.nr_measurement_blocks: int = self._project_data["Experiment metadata"].get("Number of measurement blocks", 1) 


class BaseClassNoiseLearningFigures(BaseClassNoiseLearningData):
    """
    Base class for figure plotting classes that need to load general experiment metadata.
    Classes that use this base class need to implement `_plot_figures()`.

    Extends `BaseClass`.

    Parameters
    ----------
    project_dir : Path
        Path object of the project folder.
    """

    def run(self) -> None:
        """Loads the required data and plots the figures."""
        self._load_data()
        self._plot_figures()

    def _load_data(self) -> None:
        """Loads experiment data from project folder."""
        super()._load_data()
        self._title = "{name} | Qubit pair: Q{q1}-Q{q2}"
        self._title += f"\nCircuit random instances N = {self.num_random_instances}, num_shots = {self.num_shots}"
        self._title += f"\nTotal experiment duration = {self.total_time:.1f} min, Total backend runtime = {self.backend_time:.1f} min"
        self._title += f"\n{self.timestamp}"

    def _plot_figures(self) -> None:
        """Should be implemented by specific class."""
        raise NotImplementedError(
            "`_plot_figures` is not implemented by `BaseClassFigures`. Classes extending `BaseClassFigures` need to implement this."
        )


class BaseClassExperiment:
    """
    Base class for experiments that create circuits, run them on the backend, and store all data in
    the project folder.

    Classes that use this base class need to implement `_prepare_circuits()`, and `_execute_circuits()`.
    And should (though not mandatory) extend `_create_project_data()` to save additional metadata.
    Can optionally implement `_run_analysis()` if further analysis of the data is required.

    Instead of measuring the entire readout assignment matrix, this class measures the readout
    matrix for each individual qubit and then combines them to apply the readout mitigation.
    This works under the assumption that readout crosstalk is small.
    """

    EXPERIMENT_NAME = "BaseClassExperiment"
    BASE_DIR = Path.home() / "Documents" / "QuantumInspireProjects"

    NATIVE_GATES = ["id", "z", "rz", "s", "sdg", "t", "tdg", "x", "rx", "y", "ry", "cz", "delay", "reset", "measure"]

    PROGRESSBAR_FORMAT = "{desc:>16.16}:  {percentage:3.0f}%|{bar}{r_bar}"

    def __init__(
        self,
        backend: QIBackend,
        qubit_list: list[int],
    ) -> None:
        """
        Parameters
        ----------
        backend : QIBackend
            The QI backend to use.
        qubit_list : list[int]
            An ordered list of the qubits in the quantum circuit.
        """
        # assign parameters
        self.backend = backend
        self.qubit_list = qubit_list

    def run(
        self,
        num_shots: int,
        num_random_instances: int,
        num_stitched_circuits: int = 1,
        ro_matrix_num_shots: int = 4096,
        timeout: int = 60 * 60,
        max_circuits_per_job: int = 1,
        save_data_to_disk: bool = True,
    ) -> Any:
        """
        Runs the experiment.

        Parameters
        ----------
        num_shots : int
            The number of shots used.
        num_random_instances : int
            The number of randomizations to use for methods like Pauli twirling, or
            probabilistic error amplification.
        num_stitched_circuits : int, optional
            The number of random instances that get combined into a single circuit before executing
            on the backend. Executing larger circuits with more mid-circuit measurements can reduce
            execution time compared to multiple smaller circuits. This value can not be 'too' high,
            as that will result in the circuit becoming too large for the control hardware memory.
            Defaults to 1.
        ro_matrix_num_shots : int, optional
            The number of shots used for the readout assignment matrices, by default 4096
        timeout : int, optional
            The timeout time (in seconds) used for `job.result(timeout=timeout)`, by default 1 hour.
        max_circuits_per_job : int, optional
            The maximum number of circuits to execute per job on the QuantumInspire backend,
            by default 1.
        save_data_to_disk : bool, optional
            Whether to save the data of individual jobs to disk, by default True.

        Returns
        -------
        _type_
            Returns the data given by `_execute_circuits` and `_run_analysis`, both of which get
            implemented by classes extending this base class.
        """
        # check inputs
        if num_random_instances % num_stitched_circuits != 0:
            new_value = math.ceil(num_random_instances / num_stitched_circuits) * num_stitched_circuits
            warnings.warn(
                f"""
                `num_random_instances` is not divisible by `num_stitched_circuits`.
                `num_random_instances` was provided as {num_random_instances}, while trying to combine {num_stitched_circuits} random instances into a single circuit (`num_stitched_circuits`).
                Therefore `num_random_instances` has been set to {new_value}.
                """,
                UserWarning,
            )
            num_random_instances = new_value

        # assign parameters
        self.num_shots = num_shots
        self.num_random_instances = num_random_instances
        self.num_stitched_circuits = num_stitched_circuits
        self.ro_matrix_num_shots = ro_matrix_num_shots
        self.timeout = timeout
        self.max_circuits_per_job = max_circuits_per_job
        self.save_data_to_disk = save_data_to_disk

        # initialize variables
        self._job_dirs = []
        self._backend_execution_time = 0
        self._queue_time = 0

        # perform experiment
        print("Creating project directory")
        self._create_project_dir()
        self._create_project_data()
        self._store_project_data()
        print("Creating circuits")
        self._create_readout_circuit()
        self._prepare_circuits()
        print("Executing circuits")
        output_exp = self._execute_circuits()
        self._succesfull_termination()
        output_ana = self._run_analysis()

        # don't return a tuple of outputs if one of them is None
        if output_ana is None:
            output = output_exp
        elif output_exp is None:
            output = output_ana
        else:
            output = output_exp, output_ana

        return output

    def _create_project_dir(self) -> None:
        """Records timestamp and creates project folder."""
        self.start_time = datetime.now()
        self.date = self.start_time.strftime("%Y%m%d")
        self.time = self.start_time.strftime("%H%M%S")
        self.timestamp = f"{self.date}_{self.time}"
        self.project_dir = self.BASE_DIR / self.date / f"{self.time}_{self.EXPERIMENT_NAME}"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.project_dir / f"logs_{self.timestamp}.txt"

    def _create_project_data(self) -> None:
        """Create initial metadata."""
        project_data = {
            "Experiment name": self.EXPERIMENT_NAME,
            "Timestamp": self.timestamp,
            "Experiment completed successfully": False,
            "Backend info": {
                "Backend name": self.backend.name,
                "Backend number of qubits": self.backend.num_qubits,
                "Backend maximum allowed shots": self.backend.max_shots,
            },
            "Total experiment duration [s]": 0,
            "Total experiment duration [min]": 0,
            "Execution time backend [s]": 0,
            "Execution time backend [min]": 0,
            "Queue time [s]": 0,
            "Queue time [min]": 0,
        }
        self.project_data = project_data

    def _store_project_data(self) -> None:
        """Saves the project data to a json file."""
        with (self.project_dir / f"project_data_{self.timestamp}.json").open("w") as f:
            json.dump(self.project_data, f, indent=3, cls=NumpyEncoder)

    def _create_readout_circuit(self) -> None:
        """Creates the circuit used for calibrating the readout assignment matrix."""
        base_ro_circuit = readout_matrix_calibration_circuit(
            num_qubits=len(self.qubit_list),
            parallel=getattr(self, "parallel", True),
        )
        repetitions = math.ceil(self.ro_matrix_num_shots / self.num_shots)
        self.ro_matrix_num_shots = repetitions * self.num_shots
        base_ro_circuit = stitch_circuits([base_ro_circuit] * repetitions)
        self.ro_circuit = transpile(
            base_ro_circuit,
            self.backend,
            initial_layout=self.qubit_list,
            layout_method="trivial",
            routing_method="none",
        )

    def _prepare_circuits(self) -> None:
        """Should be implemented by the specific class to create all circuits."""
        raise NotImplementedError(
            "`_prepare_circuits()` is not implemented by `BaseClassExperiment`. Classes extending `BaseClassExperiment` need to implement this."
        )

    def _execute_circuits(self) -> Any:
        """Should be implemented by the specific class to execute all circuits."""
        raise NotImplementedError(
            "`_execute_circuits()` is not implemented by `BaseClassExperiment`. Classes extending `BaseClassExperiment` need to implement this."
        )

    def _execute_jobs(
        self,
        qc_list: list[QuantumCircuit],
        job_label: str = "",
        job_metadata: dict = {},
    ) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
        """
        Executes all circuits in the list, divided over multiple jobs according to
        `max_circuits_per_job`. Adds an additional circuit to each job to perform readout error
        mitigation.
        """
        all_counts: list[list[dict]] = []
        all_raw_probs: list[list[dict]] = []
        all_probs: list[list[dict]] = []

        # make sublists for desired amount of circuits per job:
        n = self.max_circuits_per_job
        circuits: list[list[QuantumCircuit]] = [
            qc_list[idx : idx + n] for idx in range(0, len(qc_list), n)
        ]

        for job_qc_list in tqdm(circuits, desc="jobs", leave=False, position=1, bar_format=self.PROGRESSBAR_FORMAT):
            raw_counts, raw_probs, probs = self._execute_single_job(job_qc_list, job_label, job_metadata)
            all_counts.extend(raw_counts)
            all_raw_probs.extend(raw_probs)
            all_probs.extend(probs)

        return all_counts, all_raw_probs, all_probs

    def _execute_single_job(
        self,
        qc_list: list[QuantumCircuit],
        job_label: str,
        job_metadata: dict = {},
    ) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
        """Executes and performs basic analysis for a single job."""
        # add readout circuit to job
        qc_list.append(self.ro_circuit)
        # start time
        start = time.time()
        # create and execute job
        job, result = create_job_and_get_result(
            backend=self.backend,
            qc_list=qc_list,
            num_shots=self.num_shots,
            memory=True,
            timeout=self.timeout,
            log=self._log_file,
            max_retries=4,
        )
        total = time.time() - start
        job_time = get_job_execution_time(job)
        queue_time = total - job_time
        self._backend_execution_time += job_time
        self._queue_time += queue_time

        self._backend_execution_time += get_job_execution_time(job)

        raw_counts, raw_probs, ro_corrected_probs = self._process_raw_data(result, qc_list)

        # save job data to disk
        if self.save_data_to_disk:
            job_dir = self.project_dir / "job_results"
            if job_label != "":
                job_dir = job_dir / job_label
            job_record = SaveJobData(
                job=job,
                directory=job_dir,  # pyright: ignore[reportArgumentType]
                silent=True,
                store_circuit_figures=False,
            )
            for idx, circuit_data in enumerate(job.circuits_run_data):
                self._job_dirs.append(f"{job_record.project_dir.parts[-2]}_{job_record.project_dir.parts[-1]}_job_idx_{idx}__job_id_{circuit_data.job_id}")

            # save additional data to job data folder.
            with Path.open(job_record.project_dir / f"measurement_probabilities_{job_record.date_timestamp}_{job_record.job_0_timestamp}.json", "w") as f:
                store_data = {
                    "Qubits": self.qubit_list,
                    "Number of circuits stitched together": self.num_stitched_circuits,
                    "Number of shots": self.num_shots,
                    "Execution time [s]": job_time,
                    "Queue time [s]": queue_time,
                    **job_metadata,  # allows additional info to be saved
                    "Readout assignment matrices used": self._ro_matrices,
                    "Readout corrected probabilities": ro_corrected_probs,
                    "Measurement counts": raw_counts,
                }
                json.dump(store_data, f, indent=3, cls=NumpyEncoder)
                del job_record

        return raw_counts, raw_probs, ro_corrected_probs

    def _process_raw_data(
        self,
        result: Result,
        qc_list: list[QuantumCircuit],
    ) -> tuple[list[list[dict[str, int]]], list[list[dict[str, float]]], list[list[dict[str, float]]]]:
        """Processes the raw data for a single job."""
        ro_shots: list[str] = get_raw_data(qc_list[-1], result, -1)  # pyright: ignore[reportAssignmentType]
        num_bits_per_ro_subcircuit = 2 * len(self.qubit_list)
        num_ro_subcircuits = len(ro_shots[0]) // num_bits_per_ro_subcircuit
        combined_ro_shots = [
            shot[i * num_bits_per_ro_subcircuit : (i + 1) * num_bits_per_ro_subcircuit]
            for shot in ro_shots
            for i in range(0, num_ro_subcircuits)
        ]

        self._ro_matrices = analyse_readout_matrix_circuit(combined_ro_shots, self.qubit_list)
        raw_data_shots: list[list[str]]
        raw_data_shots = [get_raw_data(qc_list[idx], result, idx) for idx in range(len(qc_list)-1)] # pyright: ignore[reportAssignmentType]
        raw_data_counts = [get_multi_counts(raw_shots, len(self.qubit_list)) for raw_shots in raw_data_shots]
        raw_data_probs = [get_multi_probs(raw_counts) for raw_counts in raw_data_counts]

        ro_corrected_probs = [
            [
                apply_readout_assignment_correction(
                    raw_prob,
                    self._ro_matrices,
                    self.qubit_list,
                )
                for raw_prob in raw_probs_list
            ]
            for raw_probs_list in raw_data_probs
        ]

        return raw_data_counts, raw_data_probs, ro_corrected_probs

    def _succesfull_termination(self) -> None:
        """Adds final entries to the project data and saves it to file."""
        self.project_data["Total experiment duration [s]"] = (datetime.now() - self.start_time).total_seconds()
        self.project_data["Execution time backend [s]"] = self._backend_execution_time
        self.project_data["Total experiment duration [min]"] = self.project_data["Total experiment duration [s]"] / 60
        self.project_data["Execution time backend [min]"] = self.project_data["Execution time backend [s]"] / 60
        self.project_data["Experiment metadata"]["Job directories"] = self._job_dirs
        self.project_data["Experiment completed successfully"] = True
        self.project_data["Queue time [s]"] = self._queue_time
        self.project_data["Queue time [min]"] = self._queue_time / 60
        self._store_project_data()

    def _run_analysis(self) -> Any:
        """Can be implemented by classes extending this base class to perform additional data analysis."""
        pass
