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
import hashlib
from typing import List

import numpy as np
from qiskit import qasm3
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.states.statevector import Statevector
from scipy.sparse import coo_array, coo_matrix, csr_array, csr_matrix

from q_alchemy.initialize import q_alchemy_as_qasm, create_client, OptParams, q_alchemy_as_qasm_parallel_states


class QAlchemyInitialize(Instruction):
    """
    State preparation using Q-Alchemy API

    This class implements a state preparation gate.
    """

    def __init__(self,
                 params: Statevector | List[complex] | np.ndarray | coo_array | coo_matrix,
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
        if isinstance(params, (coo_matrix, coo_array)):
            num_qubits = int(np.ceil(np.log2(params.shape[1])))
        elif isinstance(params, (Statevector, List, np.ndarray)):
            params = np.asarray(params, dtype=complex).tolist()
            num_qubits = int(np.ceil(np.log2(len(params))))
        else:
            raise TypeError("params type not recognized")
        # Qiskit expects params to be a list of params.
        params = [params]
        if opt_params is None:
            self.opt_params = OptParams()
        elif isinstance(opt_params, OptParams):
            self.opt_params = opt_params
        else:
            self.opt_params = OptParams(**opt_params)

        self.client = create_client(self.opt_params)

        if label is None:
            label = "QAl"

        super().__init__("q-alchemy", num_qubits, 0, params=params, label=label)
        if self.opt_params.assign_data_hash:
            self.param_hash = hashlib.md5(np.asarray(self.params).tobytes()).hexdigest()
        else:
            self.param_hash = datetime.datetime.utcnow().timestamp()

    def _define(self):
        # need to unbox params again.
        qasm, summary = q_alchemy_as_qasm(self.params[0], self.opt_params, self.client, return_summary=True)
        if self.opt_params.use_qasm3:
            qc = qasm3.loads(qasm)
        else:
            qc = QuantumCircuit.from_qasm_str(qasm)
            qc.global_phase = summary["global_phase"] #n.b. this is zero if use_qasm3
        self.definition = qc

def qiskit_batch_initialize(state_vectors: list[Statevector | List[complex] | np.ndarray],
                         labels: list[str] = [],
                         opt_params: dict | OptParams | None = None) -> list[Gate]:
    """
    Submit a batch of state vectors to QAlchemy, and return a list of initialization Gates.

    Args:
        state_vectors (list[Statevector | List[complex] | np.ndarray]): A list of states.
        labels (list[str]): A list of gate labels. Defaults to [], in which case gates will be named automatically.
        opt_params (dict | OptParams | None): Optional parameters; see OptParams for details.

    Returns:
        list[Gate]: A list of labeled initialization circuits, one for each state.
    """
    params = np.asarray(state_vectors, dtype=complex).tolist()
    num_states = len(params)
    num_qubits = int(np.ceil(np.log2(len(params[0]))))
    if opt_params is None:
        opt_params = OptParams()
    elif not isinstance(opt_params, OptParams):
        opt_params = OptParams(**opt_params)

    if labels == []:
        labels = [f"QAl{i}" for i in range(num_states)]
    elif len(labels) == 1:
        labels = [labels[0]] * num_states
    elif len(labels) != num_states:
        raise ValueError(f"Number of labels ({len(labels)}) must be equal to 0, 1, or number of states ({num_states})")

    # if opt_params.assign_data_hash:
    #     param_hash = hashlib.md5(np.asarray(params).tobytes()).hexdigest()
    # else:
    #     param_hash = datetime.datetime.utcnow().timestamp()

    qasm_list, summary_list = q_alchemy_as_qasm_parallel_states(
        state_vector=params, opt_params=opt_params, num_qubits=num_qubits, client=None, return_summary=True)
    if opt_params.use_qasm3:
        qcs = [qasm3.loads(qasm) for qasm in qasm_list]
    else:
        qcs = [QuantumCircuit.from_qasm_str(qasm) for qasm in qasm_list]
        for qc, summary in zip(qcs, summary_list):
            qc.global_phase = summary["global_phase"]
    gates = [qc.to_gate(label=label) for qc, label in zip(qcs, labels)]
    return gates