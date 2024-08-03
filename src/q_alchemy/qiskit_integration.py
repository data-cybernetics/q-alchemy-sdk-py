# Copyright 2022-2023 data cybernetics ssc GmbH.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import inspect
import logging
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import httpx
import numpy as np
from httpx import HTTPTransport
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.states.statevector import Statevector

from pinexq_client.job_management import enter_jma, Job
from pinexq_client.core import MediaTypes
from pinexq_client.core.hco.upload_action_hco import UploadParameters
from pinexq_client.job_management.model import JobStates, SetTagsWorkDataParameters, WorkDataQueryParameters, \
    WorkDataFilterParameter

logging.getLogger("httpx").setLevel(logging.WARN)
LOG = logging.getLogger(__name__)


@dataclass
class OptParams:
    remove_data: bool = field(default=True)
    max_fidelity_loss: float = field(default=0.0)
    job_tags: List[str] = field(default_factory=list)
    api_key: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY"))
    host: str = field(default="jobs.api.q-alchemy.com")
    schema: str = field(default="https")
    added_headers: Dict[str, str] = field(default_factory=dict)
    isometry_scheme: str = field(default="ccd")
    unitary_scheme: str = field(default="qsd")
    job_completion_timeout_sec: int = field(default=300)
    basis_gates: List[str] = field(default_factory=lambda: ["rx", "ry", "rz", "cx"])
    image_size: Tuple[int, int] = field(default=(-1, -1))
    with_debug_data: bool = field(default=False)
    assign_data_hash: bool = field(default=True)

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


class QAlchemyInitialize(Instruction):
    """
    State preparation using Q-Alchemy API

    This class implements a state preparation gate.
    """

    def __init__(self,
                 params: Statevector | List[complex] | np.ndarray,
                 label=None,
                 opt_params: dict | OptParams | None = None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.

        opt_params: Dictionary
            max_fidelity_loss: float
                ``state`` allowed (fidelity) error for approximation
                (0<=``max_fidelity_loss``<=1). If ``max_fidelity_loss`` is not in the valid
                range, it will be ignored.

            isometry_scheme: string
                Scheme used to decompose isometries.
                Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
                Default is ``isometry_scheme='ccd'``.

            unitary_scheme: string
                Scheme used to decompose unitaries.
                Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
                Shannon decomposition).
                Default is ``unitary_scheme='qsd'``.
        """
        params = np.asarray(params, dtype=complex).tolist()
        num_qubits = int(np.ceil(np.log2(len(params))))
        if opt_params is None:
            self.opt_params = OptParams()
        elif isinstance(opt_params, OptParams):
            self.opt_params = opt_params
        else:
            self.opt_params = OptParams(**opt_params)

        headers = {"x-api-key": self.opt_params.api_key}
        headers.update(self.opt_params.added_headers)
        if label is None:
            label = "QAl"
        self.client = httpx.Client(
            base_url=f"{self.opt_params.schema}://{self.opt_params.host}",
            headers=headers,
            timeout=httpx.Timeout(
                timeout=self.opt_params.job_completion_timeout_sec + 10.0,
                connect=10.0
            ),
            transport=HTTPTransport(retries=3)
        )
        super().__init__("q-alchemy", num_qubits, 0, params=params, label=label)
        if self.opt_params.assign_data_hash:
            self.param_hash = hashlib.md5(np.asarray(self.params).tobytes()).hexdigest()
        else:
            self.param_hash = datetime.datetime.utcnow().timestamp()

    def _define(self):
        sequence_wd_tags = [
            f"Hash={self.param_hash}",
            "Source=Qiskit-Integration",
            f"ImageSize={self.opt_params.image_size}"
        ]
        sequence_wd_tags += self.opt_params.job_tags
        wd_root = enter_jma(self.client).work_data_root_link.navigate()

        existing_wd_query = wd_root.query_action.execute(WorkDataQueryParameters(
            filter=WorkDataFilterParameter(tags_by_and=sequence_wd_tags)
        ))

        if existing_wd_query.total_entities == 0:
            wd_root = enter_jma(self.client).work_data_root_link.navigate()
            wd_link = wd_root.upload_action.execute(UploadParameters(
                filename=f"{self.param_hash}.bin",
                binary=np.asarray(self.params, dtype=np.complex128).tobytes(),
                mediatype=MediaTypes.OCTET_STREAM,
            ))
            wd_link.navigate().edit_tags_action.execute(
                SetTagsWorkDataParameters(tags=sequence_wd_tags)
            )
        else:
            wd_link = existing_wd_query.workdatas[0].self_link

        job_timeout = self.opt_params.job_completion_timeout_sec * 1000
        processing_name = "convert_circuit_layers_qasm_only"
        job_parameters = dict(
            min_fidelity=1.0 - self.opt_params.max_fidelity_loss,
            basis_gates=self.opt_params.basis_gates,
        )
        if all(i > 0 for i in self.opt_params.image_size) or self.opt_params.with_debug_data:
            processing_name = "convert_circuit_layers"
            job_parameters.update(dict(
                image_shape_x=self.opt_params.image_size[0],
                image_shape_y=self.opt_params.image_size[1]
            ))
        job = (
            Job(self.client)
            .create(name=f'Execute Transformation ({datetime.datetime.now()})')
            .select_processing(function_name=processing_name)
            .configure_parameters(**job_parameters)
            .assign_input_dataslot(0, wd_link)
            .allow_output_data_deletion()
            .start()
            .wait_for_state(JobStates.completed, timeout_ms=job_timeout)
        )
        self.result_summary: dict = job.get_result()
        inner_job = job._job
        if self.result_summary["status"].startswith("OK"):
            qasm_wd = [wd for s in inner_job.output_dataslots for wd in s.assigned_workdatas if wd.name == "qasm_circuit.qasm"][0]
            if qasm_wd.size_in_bytes > 0:
                qasm: str = qasm_wd.download_link.download().decode("utf-8")
                qc = QuantumCircuit.from_qasm_str(qasm)
                self.definition = qc
            else:
                raise IOError("Q-Alchemy API call failed for unknown reasons.")
        else:
            raise IOError(f"Q-Alchemy API call failed. Reason: {self.result_summary['status']}.")
        # Clean-up now.
        if self.opt_params.remove_data and inner_job is not None:
            for od in inner_job.output_dataslots:
                for wd in od.assigned_workdatas:
                    delete_action = wd.delete_action
                    if delete_action is not None:
                        delete_action.execute()
            job.refresh().delete()
