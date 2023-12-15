import numpy as np
import qiskit
from qclib.state_preparation import LowRankInitialize
from qiskit import QuantumCircuit


def to_circuit(vectors, qubits, ranks, partitions, num_qubits, opt_params):
    """
    Return a qiskit circuit from this node.

    opt_params: Dictionary
        isometry_scheme: string
            Scheme used to decompose isometries.
            Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
            Default is ``isometry_scheme='ccd'``.

        unitary_scheme: string
            Scheme used to decompose unitaries.
            Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
            Shannon decomposition).
            Default is ``unitary_scheme='qsd'``.
    :return: the circuit to create the state
    """
    opt_params = {} if opt_params is None else opt_params
    circuit = QuantumCircuit(num_qubits)

    vector: np.ndarray
    for vector, qubits, rank, partition in zip(vectors, qubits, ranks, partitions):
        # There may be no operation necessary, if so, add identity
        if vector is None:
            for qb in qubits[::-1]:  # qiskit little-endian.
                circuit.compose(qiskit.circuit.library.IGate(), [qb], inplace=True)
        else:
            opt_params = {
                "iso_scheme": opt_params.get("isometry_scheme"),
                "unitary_scheme": opt_params.get("unitary_scheme"),
                "partition": partition,
                "lr": rank,
            }
            # Add the gate to the circuit
            gate = LowRankInitialize(list(vector), opt_params=opt_params)
            circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian.

    return circuit.reverse_bits()
