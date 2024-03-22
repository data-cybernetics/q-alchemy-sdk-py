# Copyright 2022-2023 data cybernetics ssc GmbH.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import httpx
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.states.statevector import Statevector

from hypermedia_client.job_management import enter_jma, Job
from hypermedia_client.core import MediaTypes
from hypermedia_client.core.hco.upload_action_hco import UploadParameters
from hypermedia_client.job_management.model import JobStates, SetTagsWorkDataParameters, WorkDataQueryParameters, \
    WorkDataFilterParameter

logging.getLogger("httpx").setLevel(logging.WARN)
LOG = logging.getLogger(__name__)


@dataclass
class OptParams:
    max_fidelity_loss: float = field(default=0.0)
    job_tags: List[str] = field(default_factory=list)
    api_key: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY"))
    host: str = field(default="jobs.api.q-alchemy.com")
    schema: str = field(default="https")
    added_headers: Dict[str, str] = field(default_factory=dict)
    isometry_scheme: str = field(default="ccd")
    unitary_scheme: str = field(default="qsd")
    job_completion_timeout_sec: int = field(default=300)
    basis_gates: List[str] = field(default_factory=lambda: ["u1", "u2", "u3", "cx"])
    image_size: Tuple[int, int] = field(default=(-1, -1))


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
            timeout=httpx.Timeout(timeout=self.opt_params.job_completion_timeout_sec + 10.0, connect=10.0)
        )
        super().__init__("q-alchemy", num_qubits, 0, params=params, label=label)
        self.param_hash = hashlib.md5(np.asarray(self.params).tobytes()).hexdigest()

    def _define(self):
        sequence_wd_tags = [self.param_hash, "Q-Alchemy", "Qiskit-Integration"]
        wd_root = enter_jma(self.client).work_data_root_link.navigate()

        existing_wd_query = wd_root.query_action.execute(WorkDataQueryParameters(
            filter=WorkDataFilterParameter(tags_by_and=sequence_wd_tags)
        ))

        if existing_wd_query.total_entities == 0:
            wd_root = enter_jma(self.client).work_data_root_link.navigate()
            wd_link = wd_root.upload_action.execute(UploadParameters(
                filename=f"{self.param_hash}.bin",
                binary=np.asarray(self.params).tobytes(),
                mediatype=MediaTypes.OCTET_STREAM,
            ))
            wd_link.navigate().edit_tags_action.execute(SetTagsWorkDataParameters(tags=sequence_wd_tags))
        else:
            wd_link = existing_wd_query.workdatas[0].self_link

        job_timeout = self.opt_params.job_completion_timeout_sec * 1000
        job = (
            Job(self.client)
            .create(name='Execute Transformation')
            .select_processing(processing_step='convert_circuit_layers')
            .configure_parameters(
                min_fidelity=1.0 - self.opt_params.max_fidelity_loss,
                basis_gates=self.opt_params.basis_gates,
                image_shape_x=self.opt_params.image_size[0],
                image_shape_y=self.opt_params.image_size[1]
            )
            .assign_input_dataslot(0, wd_link)
            .start()
            .wait_for_state(JobStates.completed, timeout_ms=job_timeout)
        )
        self.result_summary: dict = job.get_result()
        if self.result_summary["status"].startswith("OK"):
            inner_job = job._job
            qasm_wd = [s.assigned_workdata for s in inner_job.output_dataslots if s.assigned_workdata.name == "qasm_circuit.qasm"][0]
            if qasm_wd.size_in_bytes > 0:
                qasm: str = qasm_wd.download_link.download().decode("utf-8")
                qc = QuantumCircuit.from_qasm_str(qasm)
                self.definition = qc
            else:
                raise IOError("Q-Alchemy API call failed for unknown reasons.")
        else:
            raise IOError(f"Q-Alchemy API call failed. Reason: {self.result_summary['status']}.")
