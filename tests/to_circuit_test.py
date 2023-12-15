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
import unittest

import numpy as np
from qclib.state_preparation import LowRankInitialize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.circuit_compiler import to_circuit


class TestApiIntegration(unittest.TestCase):

    def test_integration(self):
        num_qubits = 3
        b = np.random.rand(2**num_qubits)
        b = [b / np.linalg.norm(b)]
        qubits = [[0, 1, 2]]  # Zero-Based list of qubits
        ranks = [2]
        partitions = [None]
        opt_param = {'max_fidelity_loss':0.0}

        qc = to_circuit(b, qubits, ranks, partitions, num_qubits, opt_param)

        b_test = Statevector(qc).data

        self.assertAlmostEqual(0, np.linalg.norm(b - b_test), places=3)

    def test_base_low_rank_sp(self):
        num_qubits = 3
        b = np.random.rand(2**num_qubits)
        b = [b / np.linalg.norm(b)]

        gate = LowRankInitialize(list(b[0]))
        b_test = Statevector(gate.definition).data

        self.assertAlmostEqual(0, np.linalg.norm(b - b_test), places=3)

    def test_base_low_rank_sp_partial(self):
        num_qubits = 3
        b = np.random.rand(2**num_qubits)
        b = b / np.linalg.norm(b)
        qubits = [0, 1, 2]

        gate = LowRankInitialize(list(b))

        circuit = QuantumCircuit(num_qubits)
        circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian
        circuit = circuit.reverse_bits()

        b_test = Statevector(circuit).data

        self.assertAlmostEqual(0, np.linalg.norm(b - b_test), places=3)