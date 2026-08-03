import math
import random
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy import sparse
from scipy.sparse import coo_array, coo_matrix

from q_alchemy.initialize import InitializationMethods
from q_alchemy.qiskit_integration import (
    QAlchemyInitialize,
    OptParams,
    qiskit_batch_initialize
)
from dotenv import load_dotenv

load_dotenv("../.env")

class TestQiskitIntegration(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    def assert_state_close(self, target, got, max_fidelity_loss):
        """Assert `got` prepares `target` within the *requested* fidelity budget.

        The service contract is `max_fidelity_loss`; a raw norm bound like
        ``norm(a - b) <= 0.05`` is both stricter than that contract (a loss of
        f permits a phase-aligned norm of sqrt(2 - 2*sqrt(1-f)), e.g. ~0.32
        for f=0.05) and sensitive to the physically irrelevant global phase.
        Historically it passed only because the service over-delivered
        near-exact circuits; since qtucker >= 0.2.x the loss budget is
        actually used to shorten circuits.

        The norm check is therefore phase-aligned and its bound derived from
        the requested loss. Global-phase correctness is tracked separately in
        qtucker-state-preparation issue #71 and deliberately not asserted here.
        """
        target = np.asarray(target).ravel()
        got = np.asarray(got).ravel()
        # 2% relative slack: the optimizer may legitimately land on the budget
        # boundary within numerical error.
        allowed_loss = max_fidelity_loss * 1.02 + 1e-9
        fidelity_loss = 1 - abs(np.vdot(target, got)) ** 2
        self.assertLessEqual(fidelity_loss, allowed_loss)
        phase = np.angle(np.vdot(got, target))
        aligned = np.exp(1j * phase) * got
        norm_bound = math.sqrt(2 - 2 * math.sqrt(1 - min(allowed_loss, 1.0)))
        self.assertLessEqual(np.linalg.norm(target - aligned), norm_bound + 1e-9)

    def test_fixed_complex(self):

        with open("data/test.qasm", "r") as f:
            qasm = f.read()

        qc = QuantumCircuit.from_qasm_str(qasm)
        state_vector = Statevector(qc)

        opt_params = OptParams(
        #        api_key=os.environ["Q_ALCHEMY_API_KEY"]
        )

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=opt_params
        )

        state_qiskit = Statevector(instr)

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-10) # not quite that precise?


    def test_rnd_real(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?

    def test_rnd_complex(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)
        subtest_params = [True, False]
        for use_qasm3 in subtest_params:
            with self.subTest(use_qasm3=use_qasm3):
                instr = QAlchemyInitialize(
                    params=state_vector,
                    opt_params=OptParams(
                        use_qasm3=use_qasm3,
                        #api_key="<your api key>"
                    )
                )
                circuit_qiskit = instr.definition

                state_qiskit = Statevector(circuit_qiskit).data

                self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
                self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?

    def test_large_complex(self):

        n_qubits = 12
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                max_fidelity_loss=0.05
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data # 16 is too large!

        self.assert_state_close(state_vector, state_qiskit, max_fidelity_loss=0.05)

    def test_batch_complex(self):
        n_qubits = 4
        n_states = 4
        state_vectors = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j] * n_states
        state_vectors = [sv / np.linalg.norm(sv) for sv in state_vectors]

        subtest_params = [True, False]

        for use_qasm3 in subtest_params:
            with self.subTest(use_qasm3=use_qasm3):
                gate_list = qiskit_batch_initialize(state_vectors=state_vectors, opt_params=OptParams(
                    use_qasm3=use_qasm3,
                    initialization_method=InitializationMethods.HIERARCHICAL_TUCKER,
                    max_fidelity_loss=0.05
                    # api_key="<your api key>"
                ))
                qcs = [QuantumCircuit(n_qubits) for gate in gate_list]
                for gate, qc in zip(gate_list, qcs):
                    qc.append(gate, range(n_qubits))

                qiskit_states = [Statevector(circuit).data for circuit in qcs]

                for init_state, qiskit_state in zip(state_vectors, qiskit_states):
                    self.assert_state_close(init_state, qiskit_state, max_fidelity_loss=0.05)

                for qc in qcs:
                    print(qc)

    def test_coo_matrix(self):
        n_qubits = 8
        n_terms = 7
        assert n_terms <= 2 ** n_qubits
        coo_data = np.random.rand(n_terms) + np.random.rand(n_terms) * 1j
        coo_data = coo_data / np.linalg.norm(coo_data)
        coo_rows = np.array([0 for i in range(n_terms)])
        coo_cols = np.array(random.sample(range(2 ** n_qubits), n_terms))
        coo_state = coo_matrix((coo_data, (coo_rows, coo_cols)), shape=(1, 2 ** n_qubits))

        instr = QAlchemyInitialize(
            params=coo_state,
            opt_params=OptParams(
                max_fidelity_loss=0.01
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition
        state_vector = coo_state.toarray()
        state_qiskit = Statevector(circuit_qiskit).data # 16 is too large!

        self.assert_state_close(state_vector, state_qiskit, max_fidelity_loss=0.01)

    def test_coo_array(self):
        n_qubits = 8
        n_terms = 7
        assert n_terms <= 2 ** n_qubits
        coo_data = np.random.rand(n_terms) + np.random.rand(n_terms) * 1j
        coo_data = coo_data / np.linalg.norm(coo_data)
        coo_rows = np.array([0 for i in range(n_terms)])
        coo_cols = np.array(random.sample(range(2 ** n_qubits), n_terms))
        coo_state = coo_array((coo_data, (coo_rows, coo_cols)), shape=(1, 2 ** n_qubits))
        instr = QAlchemyInitialize(
            params=coo_state,
            opt_params=OptParams(
                max_fidelity_loss=0.01
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition
        state_vector = coo_state.toarray()
        state_qiskit = Statevector(circuit_qiskit).data # 16 is too large!

        self.assert_state_close(state_vector, state_qiskit, max_fidelity_loss=0.01)
    
    def test_big_coo(self):
        n_qubits = 8
        n_terms = 7
        assert n_terms <= 2 ** n_qubits
        coo_data = np.random.rand(n_terms) + np.random.rand(n_terms) * 1j
        coo_data = coo_data / np.linalg.norm(coo_data)
        coo_rows = np.array([0 for i in range(n_terms)])
        coo_cols = np.array(random.sample(range(2 ** n_qubits), n_terms))
        coo_state = coo_matrix((coo_data, (coo_rows, coo_cols)), shape=(1, 2 ** n_qubits))
        coo_states = sparse.vstack([coo_state for i in range(4)])

        gate_list = qiskit_batch_initialize(state_vectors=coo_states, opt_params=OptParams(
            initialization_method=InitializationMethods.HIERARCHICAL_TUCKER,
            max_fidelity_loss=0.05
            # api_key="<your api key>"
        ))
        qcs = [QuantumCircuit(n_qubits) for gate in gate_list]
        for gate, qc in zip(gate_list, qcs):
            qc.append(gate, range(n_qubits))

        qiskit_states = [Statevector(circuit).data for circuit in qcs]
        state_vectors = coo_states.toarray()
        for init_state, qiskit_state in zip(state_vectors, qiskit_states):
            self.assert_state_close(init_state, qiskit_state, max_fidelity_loss=0.05)

        for qc in qcs:
            print(qc)

    def test_batch_coo(self):
        n_qubits = 8
        n_terms = 7
        assert n_terms <= 2 ** n_qubits
        coo_data = np.random.rand(n_terms) + np.random.rand(n_terms) * 1j
        coo_data = coo_data / np.linalg.norm(coo_data)
        coo_rows = np.array([0 for i in range(n_terms)])
        coo_cols = np.array(random.sample(range(2 ** n_qubits), n_terms))
        coo_state = coo_matrix((coo_data, (coo_rows, coo_cols)), shape=(1, 2 ** n_qubits))
        coo_states = [coo_state for i in range(4)]

        gate_list = qiskit_batch_initialize(state_vectors=coo_states, opt_params=OptParams(
            initialization_method=InitializationMethods.HIERARCHICAL_TUCKER,
            max_fidelity_loss=0.05
            # api_key="<your api key>"
        ))
        qcs = [QuantumCircuit(n_qubits) for gate in gate_list]
        for gate, qc in zip(gate_list, qcs):
            qc.append(gate, range(n_qubits))

        qiskit_states = [Statevector(circuit).data for circuit in qcs]
        state_vectors = [cs.toarray() for cs in coo_states]
        for init_state, qiskit_state in zip(state_vectors, qiskit_states):
            self.assert_state_close(init_state, qiskit_state, max_fidelity_loss=0.05)

        for qc in qcs:
            print(qc)

if __name__ == '__main__':
    unittest.main()
