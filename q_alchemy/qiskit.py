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

import datetime
import json

from dataclasses import dataclass, asdict
import time

import numpy as np
from qclib.util import get_state

from qclib.gates.initialize import Initialize
from qiskit import QuantumCircuit

from .client import Client, JobConfigWrapper, StateVector
from .models import QAlchemyError


@dataclass
class _OptParams:
    def __init__(self, opt_params):
        if opt_params is None:
            self.max_fidelity_loss = 0.0
            self.job_tags = None
            self.api_key = None
            self.host = None
            self.schema = None
            self.added_headers = None
            self.isometry_scheme = "ccd"
            self.unitary_scheme = "qsd"
            self.job_completion_timeout_sec = 5 * 60
            self.use_result_after_sec = self.job_completion_timeout_sec
        else:
            self.max_fidelity_loss = 0.0 if opt_params.get("max_fidelity_loss") is None \
                else opt_params.get("max_fidelity_loss")

            self.job_tags = None if opt_params.get("job_tags") is None \
                else opt_params.get("job_tags")

            self.api_key = None if opt_params.get("api_key") is None \
                else opt_params.get("api_key")

            self.host = None if opt_params.get("host") is None \
                else opt_params.get("host")

            self.schema = None if opt_params.get("schema") is None \
                else opt_params.get("schema")

            self.added_headers = None if opt_params.get("added_headers") is None \
                else opt_params.get("added_headers")

            self.isometry_scheme = "ccd" if opt_params.get("iso_scheme") is None else \
                opt_params.get("iso_scheme")

            self.unitary_scheme = "qsd" if opt_params.get("unitary_scheme") is None else \
                opt_params.get("unitary_scheme")

            self.job_completion_timeout_sec = 5 * 60 \
                if opt_params.get("job_completion_timeout_sec") is None else \
                int(opt_params.get("job_completion_timeout_sec"))

            self.use_result_after_sec = self.job_completion_timeout_sec \
                if opt_params.get("use_result_after_sec") is None else \
                int(opt_params.get("use_result_after_sec"))

        if self.max_fidelity_loss < 0 or self.max_fidelity_loss > 1:
            self.max_fidelity_loss = 0.0


class QAlchemyInitialize(Initialize):
    """
    State preparation using Q-Alchemy API

    This class implements a state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
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
        self._name = "q-alchemy"
        self._get_num_qubits(params)
        self.node = None
        self.opt_params = _OptParams(opt_params)

        if label is None:
            self._label = "QAl"

        client = Client(
            api_key=self.opt_params.api_key,
            host=self.opt_params.host,
            schema=self.opt_params.schema,
            added_headers=self.opt_params.added_headers
        )

        job = client.get_jobs_root().update().create_job()

        # Configure the job
        job_config: JobConfigWrapper = job.get_config().create_config()
        job_config.with_max_fidelity_loss(self.opt_params.max_fidelity_loss)
        tags = ["QAlchemy-Qiskit", f"{self.num_qubits}qb", self.label]
        if self.opt_params.job_tags is not None:
            tags += self.opt_params.job_tags
        job_config.with_tags(*tags)
        while not job_config.upload():
            time.sleep(5)

        # Upload State
        job_state_vector: StateVector = job.get_state_vector()
        while not job_state_vector.upload_vector(np.asarray(params)):
            time.sleep(5)

        # Execute job@Q-Alchemy
        job.update()
        while not job.schedule():
            time.sleep(5)
        self.job = job

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        start = datetime.datetime.now()
        use_preliminary_result = False
        while not self.job.update().has_stopped:
            time.sleep(5)
            time_passed_sec = (datetime.datetime.now() - start).total_seconds()
            if time_passed_sec > self.opt_params.job_completion_timeout_sec:
                raise TimeoutError(
                    f"Waiting for the job to finish exceeded the "
                    f"timeout of {self.opt_params.job_completion_timeout_sec}!"
                )
            elif time_passed_sec > self.opt_params.use_result_after_sec:
                use_preliminary_result = True
                self.job.cancel()
                break

        if self.job.is_success or use_preliminary_result:
            return self.job.get_result().update().get_best_node().to_circuit(opt_params=asdict(self.opt_params))
        else:
            raise QAlchemyError(json.dumps(self.job.error))

    def get_node(self):
        # make sure that the job is done!
        circ = self.definition
        return self.job.get_result().get_best_node().to_node()

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a QAlchemyInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                QAlchemyInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(QAlchemyInitialize(state, opt_params=opt_params), qubits)


def fidelity(input_state: np.ndarray, transpiled_circuit: QuantumCircuit):
    ket = get_state(transpiled_circuit)
    bra = np.conj(input_state)
    return np.abs(bra.dot(ket))**2
