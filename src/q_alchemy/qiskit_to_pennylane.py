import logging
import re
import sys
from typing import Tuple

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qiskit._accelerate.quantum_circuit import CircuitInstruction
from qiskit.circuit import Instruction, Qubit

from q_alchemy.qiskit_integration import QAlchemyInitialize, OptParams

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


def extract_gates(inst_str: str):
    name_match = re.findall(r'name=\'(\w+)\'', inst_str)
    params_match = re.findall(r'params=\[(.*?)\]', inst_str)
    return name_match, params_match


def extract_qubits(inst_str: str):
    qubit_matches = re.findall(r'\),\s(\d)', inst_str)
    if qubit_matches:
        return qubit_matches
    else:
        return None


class QAlchemyStatePreparation(Operation):
    def __init__(self, state_vector, wires, id=None, **kwargs):

        # Right now, only the basis gates as given below can be set.
        if "basis_gates" in kwargs:
            raise Warning(f"Basis Gates cannot be set currently. The inpot will be ignored.")
        opt_params = OptParams.from_dict(kwargs)
        opt_params.basis_gates = ["id", "rx", "ry", "rz", "cx"]
        opt_params.job_tags += ["Source=PennyLane-Integration"]
        self._hyperparameters = {
            "opt_params": opt_params
        }
        # check if the `state_vector` param is batched
        batched = len(qml.math.shape(state_vector)) > 1

        state_batch = state_vector if batched else [state_vector]

        # apply checks to each state vector in the batch
        for i, state in enumerate(state_batch):
            shape = qml.math.shape(state)

            if len(shape) != 1:
                raise ValueError(
                    f"State vectors must be one-dimensional; vector {i} has shape {shape}."
                )

            n_amplitudes = shape[0]
            if n_amplitudes != 2 ** len(qml.wires.Wires(wires)):
                raise ValueError(
                    f"State vectors must be of length {2 ** len(wires)} or less; vector {i} has length {n_amplitudes}."
                )

            if not qml.math.is_abstract(state):
                norm = qml.math.sum(qml.math.abs(state) ** 2)
                if not qml.math.allclose(norm, 1.0, atol=1e-3):
                    raise ValueError(
                        f"State vectors have to be of norm 1.0, vector {i} has norm {norm}"
                    )

        super().__init__(state_vector, wires=wires, id=id)
        
    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(state_vector, wires, **hyperparameters):  # pylint: disable=arguments-differ
        opt_params = hyperparameters.get("opt_params", OptParams(basis_gates=["id", "rx", "ry", "rz", "cx"]))
        if len(qml.math.shape(state_vector)) > 1:
            raise ValueError(
                "Broadcasting with QAlchemyStatePreparation is not supported. Please use the "
                "qml.transforms.broadcast_expand transform to use broadcasting with "
                "QAlchemyStatePreparation."
            )

        # change ordering of wires, since original code
        # was written for IBM machines
        wires_reverse = wires[::-1]

        op_list = []
        
        init: Instruction = QAlchemyInitialize(
            np.array(state_vector, dtype=np.complex128).tolist(),
            opt_params=opt_params
        )
        inst = init.definition.data
        num_qubits = init.num_qubits

        instruction: CircuitInstruction
        for instruction in inst:
            gate_name = instruction.operation.name
            gate_param = instruction.operation.params
            wires = [num_qubits - (q._index + 1) for q in instruction.qubits]

            if gate_name == 'rz':
                op_list.append(qml.RZ(gate_param[0], wires=wires))
            elif gate_name == 'cx':
                op_list.append(qml.CNOT(wires=[num_qubits-1-int(qubits[i][0]), num_qubits-1-int(qubits[i][1])]))
            elif gate_name == 'ry':
                op_list.append(qml.RY(gate_param[0], wires=wires))
            elif gate_name == 'rx':
                op_list.append(qml.RX(gate_param[0], wires=wires))
        return op_list
