import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.opflow import (
    Z,
    I,
    StateFn,
    TensoredOp,
)
from qiskit.quantum_info import Statevector

from q_alchemy.initialize import q_alchemy_as_qasm


def calculate_norm(qc: QuantumCircuit, nb: int, nl: int, na: int) -> float:
    """Calculates the value of the euclidean norm of the solution.

    Args:
        qc: The quantum circuit preparing the solution x to the system.

    Returns:
        The value of the euclidean norm of the solution.
    """

    # Create the Operators Zero and One
    zero_op = (I + Z) / 2
    one_op = (I - Z) / 2

    # Norm observable
    observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ (I ^ nb)
    norm_2 = (~StateFn(observable) @ StateFn(qc)).eval()

    return np.real(np.sqrt(norm_2))


def get_solution_vector(solution, length):
    raw_solution_vector = Statevector(solution.state)
    numq = solution.state.num_qubits
    n = 2**(numq-1)
    print(raw_solution_vector)
    solution_vector = Statevector(solution.state).data[n:n+length].real
    print(solution_vector)
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)


import re


def ibm_decompose(qc):
    state_inst = qc.data[0]
    qubit = str(state_inst[1][0])
    name = re.findall(r"'([^']*)'", qubit)[0]
    qr = QuantumRegister(3, name)
    qc_raw = QuantumCircuit(qr)
    qc_raw.data = [state_inst]
    qc = transpile(qc_raw, basis_gates=["id", "rx", "ry", "rz", "cx"])
    return qc


def q_alchemy_prep(b, fid_loss):
    sp_qasm = q_alchemy_as_qasm(b, max_fidelity_loss=fid_loss, basis_gates=["id", "rx", "ry", "rz", "cx"])
    qc = transpile(QuantumCircuit.from_qasm_str(sp_qasm), optimization_level=3, basis_gates=["id", "rx", "ry", "rz", "cx"])
    return qc


def circuit_depth_generator(qasm_list):
    depth_list = []
    for qasm in qasm_list:
        qc = transpile(
            QuantumCircuit.from_qasm_str(qasm),
            optimization_level=3,
            basis_gates=["id", "rx", "ry", "rz", "cx"]
        )
        depth = qc.depth()
        depth_list.append(depth)
    return depth_list


def answer_comparison(x1, x2):
    dot_product = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)

    cosine_similarity = dot_product / (norm1 * norm2)
    angle_diff = np.arccos(cosine_similarity)

    diff = (x1 - x2)
    norm_diff = np.linalg.norm(diff) / np.linalg.norm(x2)
    return angle_diff, norm_diff
