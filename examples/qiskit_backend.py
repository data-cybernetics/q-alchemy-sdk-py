"""Use the hosted Q-Alchemy simulator as a Qiskit BackendV2 (and from PennyLane).

Set Q_ALCHEMY_API_KEY in the environment or the repository's .env file.
Run the Qiskit demo from the repository root with the general examples extra.

    uv run --locked --no-dev --extra examples python examples/qiskit_backend.py

Add the PennyLane extra to run both demos in the same environment.
"""

import os
from importlib.util import find_spec
from pathlib import Path

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile

from q_alchemy import QAlchemyBackend, QAlchemyProvider


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
    if find_spec("pennylane_qiskit") is None:
        print("\nInstall the PennyLane extra to run both demos.")
        print(
            "uv run --locked --no-dev --extra examples --extra qiskit --extra pennylane "
            "python examples/qiskit_backend.py"
        )
        return

    import pennylane as qml

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
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    assert os.getenv("Q_ALCHEMY_API_KEY"), "Set Q_ALCHEMY_API_KEY (e.g. in the repository .env)"
    qiskit_demo()
    pennylane_demo()
