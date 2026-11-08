"""
General utility functions.

Authors: Jan Hemink
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from qi_utilities.utility_functions.data_handling import StoreProjectRecord
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.result import Result
    from qiskit_quantuminspire.qi_backend import QIBackend
    from qiskit_quantuminspire.qi_jobs import QIJob


def stitch_circuits(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    """
    Combines a list of `QuantumCircuits` into a single `QuantumCircuit`.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of `QuantumCircuits` to be combined. All circuits need to have the same number of
        qubits and classical bits.

    Returns
    -------
    QuantumCircuit
        Combination of all circuits in the list, will copy name and metadata from `circuits[0]`.
    """
    num_circuits = len(circuits)
    if num_circuits == 1:
        return circuits[0]
    num_qubits = circuits[0].num_qubits
    num_clbits = sum([circuit.num_clbits for circuit in circuits])
    qc = QuantumCircuit(
        num_qubits, num_clbits, name=circuits[0].name, metadata=circuits[0].metadata
    )
    clbits_used = 0
    for idx, circuit in enumerate(circuits):
        qc.compose(
            circuit,
            clbits=np.arange(circuit.num_clbits) + clbits_used,
            inplace=True,
        )
        clbits_used += circuit.num_clbits
    return qc


def get_job_result(
    job: QIJob,
    timeout: int | None,
    max_retries: int = 6,
    log: Path | None = None,
) -> Result:
    """
    Tries to obtain results from job. Prevents process from crashing immediately when an exception
    is raised, wait some time and try again. This way temporary disconnections won't crash an
    experiment. If a log file is provided, will store errors in the log file.
    """
    for i in range(max_retries):
        try:
            result = job.result(timeout=timeout)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as f:
                    f.write(f"{datetime.now()}: {type(e)}: {e}\n")

            # wait for a bit and try again
            sleep(10 + 60 * (i))
            if i == max_retries - 1:
                raise e
    return result  # pyright: ignore[reportPossiblyUnboundVariable]


def create_job_and_get_result(
    backend: QIBackend,
    qc_list: list[QuantumCircuit],
    num_shots: int,
    memory: bool = False,
    timeout: int = 60 * 60,
    max_retries: int = 5,
    log: Path | None = None,
) -> tuple[QIJob, Result]:
    """
    Run a job on a QI backend and get the results. Prevents process from crashing immediately when
    an exception is raised, wait some time and try again. This way temporary disconnections won't
    crash an experiment.
    If a log file is provided, will store errors in the log file.

    Parameters
    ----------
    backend : QIBackend
        The QI backend.
    qc_list : list[QuantumCircuit]
        A list of the quantum circuits to execute.
    num_shots : int
        The number of shots to use.
    memory : bool, optional
        Passed as kwarg to `backend.run`. When set to True, includes the raw data in the results.
        Defaults to False.
    timeout : int, optional
        The timeout used for `job.result(timeout=timeout)`, by default 1 hour.
    max_retries : int, optional
        The maximum number of attempts to retry executing the job or getting the results,
        by default 6.
    log : Path | None, optional
        If provided, will store raised exceptions into this file, by default None.

    Returns
    -------
    tuple[QIJob, Result]
        The job and result.
    """
    job: QIJob = _run_func_try_except(
        func=backend.run,
        max_retries=max_retries,
        log=log,
        args=[qc_list],
        kwargs={
            "shots": num_shots,
            "memory": memory,
        },
    )

    result: Result = _run_func_try_except(
        func=job.result,
        max_retries=max_retries,
        log=log,
        kwargs={
            "timeout": timeout,
        },
    )

    return job, result


R = TypeVar("R")


def _run_func_try_except(
    func: Callable[..., R],
    max_retries: int = 5,
    log: Path | None = None,
    args: list = [],
    kwargs: dict = {},
) -> R:
    """
    Executes provided function inside a try except block. When an error occurs the function is tried
    again after a small delay. The error is logged in the log file (if provided).
    """
    i = 0
    while True:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # if a log file was provided, save the error.
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as f:
                    f.write(f"{datetime.now()}: {type(e)}: {e}\n")

            # if reached max retries raise the error
            if i > max_retries - 1:
                raise e

            # wait for a bit and try again
            sleep(10 + 30 * (i))
            i += 1


def get_job_execution_time(job: QIJob) -> float:
    """
    Sums execution time of all circuits executed in a QuantumInspire job.

    Parameters
    ----------
    job : QIJob
        The already executed QuantumInspire job.

    Returns
    -------
    float
        The execution time in seconds of all circuits in the QuantumInspire job.
    """
    execution_time = 0.0
    for circuit_data in job.circuits_run_data:
        execution_time += getattr(circuit_data.results, "execution_time_in_seconds", 0)
    return execution_time


def _check_pauli_str(string: str) -> None:
    # checks if pauli strings contain other characters
    allowed_values = r"[I,X,Y,Z]+"
    if not re.fullmatch(allowed_values, string):
        raise ValueError(
            "Unexpected character encountered in pauli strings. Please make sure that they only consist of combinations of 'I', 'X', 'Y', and 'Z'."
        )


def _paulis_commute(
    pauli_str_1: str,
    pauli_str_2: str,
) -> bool:
    """
    Function that performs the actual commutation check for two pauli strings.
    See `paulis_commute` for more information.
    """
    commute = True
    for idx in range(len(pauli_str_1)):
        pauli_1 = pauli_str_1[idx]
        pauli_2 = pauli_str_2[idx]
        if (pauli_1 == "I") or (pauli_2 == "I") or (pauli_1 == pauli_2):
            continue
        commute = not commute
    return commute


def paulis_commute(
    pauli_str_1: str,
    pauli_str_2: str,
) -> bool:
    """
    Returns whether two pauli operators commute based on their pauli strings.

    It does this by checking if an odd or even number of single qubit pauli's in the string
    commute or not. If even the operators commute, if odd they do not.

    This works because pauli operators consist of single qubit pauli's, which can either
    commute or anti-commute with other single qubit pauli's.

    Parameters
    ----------
    pauli_str_1 : str
        Pauli string representation of pauli operator 1.
        Can only consist of 'I', 'X', 'Y', and 'Z'.
    pauli_str_2 : str
        Pauli string representation of pauli operator 2.
        Can only consist of 'I', 'X', 'Y', and 'Z'.

    Returns
    -------
    bool
        Whether the two pauli operators commute or not.

    See Also
    --------
    `_paulis_commute` for the actual implementation of checking the commutation.
    """
    if len(pauli_str_1) != len(pauli_str_2):
        raise ValueError(
            "Both pauli strings (pauli_str_1 and pauli_str_2) should have the same length"
        )
    # checks if pauli strings contain other characters
    _check_pauli_str(pauli_str_1 + pauli_str_2)
    return _paulis_commute(pauli_str_1, pauli_str_2)


def _convert_to_z_basis(observable: str) -> str:
    """Replaces 'X' and 'Y' in a string by 'Z'."""
    return observable.replace("X", "Z").replace("Y", "Z")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy objects to standard python objects."""

    def default(self, obj):  # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def get_path_from_timestamp(
    timestamp: str, base_dir: Path = Path.home() / "Documents" / "QuantumInspireProjects"
) -> Path:
    """
    Returns the `Path` object for a project folder from the `date` and `time` of the project's timestamp.

    Parameters
    ----------
    timestamp : str
            The timestamp at which the experiment was performed, in string format: 'yyyymmdd_HHMMSS',
            e.g. '20260219_155500' (February 2nd 2026 15:55:00).
    base_dir : Path, optional
        The base directory used to store all project folders, by default Path.home()/"Documents"/"QuantumInspireProjects".

    Returns
    -------
    Path
        The path to the project folder corresponding to `date` and `time`.

    Raises
    ------
    FileNotFoundError
        When no project folder with `date` and `time` can be found in `base_dir`.
    """
    date, time = timestamp.split("_")
    date_path = base_dir / date
    if not date_path.is_dir():
        raise FileNotFoundError(f"Could not find the folder for date {date} at {date_path}")
    for folder in date_path.iterdir():
        if folder.name.split("_")[0] == time:
            return folder
    raise FileNotFoundError(f"Could not find a project folder starting with {time} in {date_path}")


def align_cz_gates(
    qc: QuantumCircuit,
    parallel: bool = True,
) -> QuantumCircuit:
    """
    Aligns all CZ gates in the circuit, either in parallel or sequentially.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit.
    parallel : bool, optional
        If True, aligns CZ gates to happen at the same time. Else perform them sequentially.
        Defaults to True.

    Returns
    -------
    QuantumCircuit
        The resulting quantum circuit.
    """
    output_qc = qc.copy_empty_like()
    instruction_list = qc.data.copy()

    active_qubits = set(qubit for op in qc for qubit in op.qubits) # if circuit is already transpiled there can also be idle qubits 

    cz_gates = []
    idx = 0
    blocked_qubits = set()

    # loop over instructions until the list is empty
    while instruction_list:
        # if we checked all instructions or all qubits are  blocked, add the current CZ gate layer to the circuit with barriers.
        if (idx > len(instruction_list) - 1) or blocked_qubits.issuperset(active_qubits):
            output_qc.barrier()
            for i, cz_gate in enumerate(cz_gates):
                output_qc._data.append(cz_gate)

                if not parallel:
                    output_qc.barrier()

            if parallel:
                output_qc.barrier()

            idx = 0
            cz_gates = []
            blocked_qubits = set()

        # if the next operation is on blocked qubits move on to the next
        if not blocked_qubits.isdisjoint(instruction_list[idx].qubits):
            blocked_qubits = blocked_qubits.union(instruction_list[idx].qubits) # in case it is a multi qubit operation
            idx += 1
            continue

        # get the next instruction
        instruction = instruction_list.pop(idx)

        # if not a CZ gate, add it to the output circuit
        if instruction.name != "cz":
            output_qc._data.append(instruction)
            continue

        # if it is a CZ gate, add it to the waiting list and block the qubits
        cz_gates.append(instruction)
        blocked_qubits = blocked_qubits.union(instruction.qubits)  # mark these qubits as blocked

    return output_qc


def get_qubit_pairs(qc: QuantumCircuit) -> list[list[int]]:
    """
    Return the qubit pairs that occur as a CZ gate in the circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit.

    Returns
    -------
    list[list[int]]
        A list of the qubit pairs.
    """
    qubit_pairs = []
    for instruction in qc:
        if instruction.name != "cz":
            continue

        qubits = [qubit._index for qubit in instruction.qubits]
        reverse = qubits.copy()
        reverse.reverse()

        if qubits in qubit_pairs or reverse in qubit_pairs:
            continue

        qubit_pairs.append(qubits)
    return qubit_pairs


class SaveJobData(StoreProjectRecord):
    def create_project_directory(self, job: QIJob, directory: str = None):
        """
        Creates a new project folder. Does not create a parent folder with the date.
        Besides that, functionality is exactly the same as in `qi_utilities`.

        Parameters
        ----------
        job : QIJob
            The user already-submitted job (project) object.
        directory : str
            Specifies the directory path in which the project record is to be
            stored.
            For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        timestamp_utc = job.circuits_run_data[0].results.created_on # actually when the job finished, not when created
        timestamp = timestamp_utc.astimezone()
        self.date_timestamp = timestamp.strftime("%Y%m%d")
        self.job_0_timestamp = timestamp.strftime("%H%M%S")

        self.project_name = job.program_name
        if directory is not None:
            self.base_dir = Path(directory)
        else:
            self.base_dir = Path.home() / "Documents" / "QuantumInspireProjects"
        self.project_dir = self.base_dir / f"{self.job_0_timestamp}_{self.project_name}"
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def create_job_directory(self, job: QIJob, job_idx: int, directory: str | None = None):
        """
        Creates new directories for each circuit in the job. Does not create a parent folder with the date.
        Besides that, functionality is exactly the same as in `qi_utilities`.

        Parameters
        ----------
        job : QIJob
            The user already-submitted job (project) object.
        job_idx : int
            The job index for all jobs contained within the project.
            While a project may contain a certain number of jobs, e.g. N,
            it is generally true that the execution of all these jobs
            in the Quantum Inspire platform is not sequential with respect
            to the order with which those jobs were created.
            Therefore, job_idx is being utilized for clarity when storing
            the data, so that it follows the sequence with which the jobs were
            created.
        directory : str
            Specifies the directory path in which the project record is to be
            stored.
            For no specified path, it defaults to "Documents/QuantumInspireProjects".
        """
        timestamp_utc = job.circuits_run_data[job_idx].results.created_on   # actually when the job finished, not when created
        timestamp = timestamp_utc.astimezone()
        self.date_timestamp = timestamp.strftime("%Y%m%d")
        self.job_timestamp = timestamp.strftime("%H%M%S")
        self.job_id = job.circuits_run_data[job_idx].results.job_id # also job.circuits_run_data[job_idx].job_id works

        if directory is not None:
            self.base_dir = Path(directory)
        else:
            self.base_dir = Path.home() / "Documents" / "QuantumInspireProjects"
        self.job_dir = (
            self.base_dir
            / f"{self.job_0_timestamp}_{self.project_name}"
            / f"job_idx_{job_idx}__job_id_{self.job_id}"
        )
        self.job_dir.mkdir(parents=True, exist_ok=True)
