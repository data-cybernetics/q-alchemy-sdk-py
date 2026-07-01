"""Use the hosted Q-Alchemy simulator as a Qiskit BackendV2 (and from PennyLane).

Run it with an API key available (e.g. Q_ALCHEMY_API_KEY in ../.env):

    python examples/qiskit_backend.py

Dependencies: the `examples` extra installs everything needed to run this
(Qiskit + the PennyLane-Qiskit plugin)::

    pip install "q-alchemy-sdk-py[examples]"

The Qiskit section alone only needs `[qiskit]`; the PennyLane section needs
pennylane + the PennyLane-Qiskit plugin (`[pennylane]` or `[examples]`).
"""

import os

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile

from q_alchemy import QAlchemyBackend, QAlchemyProvider

load_dotenv("../.env")
assert os.getenv("Q_ALCHEMY_API_KEY"), "Set Q_ALCHEMY_API_KEY (e.g. in ../.env)"


def qiskit_demo() -> None:
    print("== Qiskit BackendV2 ==")
    # IBM-style discovery, or just QAlchemyBackend() directly.
    backend = QAlchemyProvider().get_backend()
    print(f"backend: {backend.name} | tier: {backend.simulator.tier}")

    # Sampling: measure -> counts.
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    job = backend.run(transpile(bell, backend), shots=4096, seed_simulator=11)
    print("counts:", job.result().get_counts())

    # Exact sparse state-vector for a measurement-free circuit.
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    result = backend.run(ghz, save_sparse_statevector=True, sparse_index_format="bitstring").result()
    sv = result.data(0)["sparse_statevector"]
    print(f"sparse statevector: nnz={sv['nnz']} indices={sv['indices']}")


def pennylane_demo() -> None:
    try:
        import pennylane as qml
    except ImportError:
        print('\n== PennyLane == (skipped: install "q-alchemy-sdk-py[examples]" or "[pennylane]")')
        return

    print("\n== PennyLane via the PennyLane-Qiskit plugin ==")
    # Any Qiskit BackendV2 can back a PennyLane device through qiskit.remote.
    dev = qml.device("qiskit.remote", wires=2, backend=QAlchemyBackend(), shots=4096)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.counts()

    print("counts:", circuit())


if __name__ == "__main__":
    qiskit_demo()
    pennylane_demo()
