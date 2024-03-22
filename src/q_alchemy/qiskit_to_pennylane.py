import logging
import re
import sys

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qiskit.circuit import Instruction

sys.path.append('..')
from q_alchemy.qiskit_integration import QAlchemyInitialize, OptParams

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


class convert_qiskit(Operation):
    def __init__(self, state_vector, wires, id=None):
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
    def compute_decomposition(state_vector, wires):  # pylint: disable=arguments-differ
        if len(qml.math.shape(state_vector)) > 1:
            raise ValueError(
                "Broadcasting with MottonenStatePreparation is not supported. Please use the "
                "qml.transforms.broadcast_expand transform to use broadcasting with "
                "MottonenStatePreparation."
            )
        def extract_gates(inst_str:str):
            name_match = re.findall(r'name=\'(\w+)\'', inst_str)
            params_match = re.findall(r'params=\[(.*?)\]', inst_str)
            return name_match, params_match
    
        def extract_qubits(inst_str:str):
            qubit_matches = re.findall(r'\),\s(\d)', inst_str)
            if qubit_matches:
                return qubit_matches
            else:
                return None

        # change ordering of wires, since original code
        # was written for IBM machines
        wires_reverse = wires[::-1]

        op_list = []
        
        init: Instruction = QAlchemyInitialize(
            np.array(state_vector),
            opt_params=OptParams(
                max_fidelity_loss=0.0,
                basis_gates=["id", "rx", "ry", "rz", "cx"]
            )
        )
        inst = init.definition
        gates = []
        params = []
        num_qubits = init.num_qubits
        qubits = []
        
        for i in range(len(inst)):
            gate, param = extract_gates(str(inst[i][0]))
            qubit = extract_qubits(str(inst[i][1]))
            #print(f'adding gate {gate[0]}')
            gates.append(gate[0])
            params.append(param[0])
            qubits.append(qubit)
            print(op_list)

            if gates[i] == 'rz':
                op_list.append(qml.RZ(float(params[i]), wires=num_qubits-1-int(qubits[i][0])))
            elif gates[i] == 'cx':
                op_list.append(qml.CNOT(wires=[num_qubits-1-int(qubits[i][0]), num_qubits-1-int(qubits[i][1])]))
            elif gates[i] == 'ry':
                op_list.append(qml.RY(float(params[i]), wires=num_qubits-1-int(qubits[i][0])))
            elif gates[i] == 'rx':
                op_list.append(qml.RX(float(params[i]), wires=num_qubits-1-int(qubits[i][0])))
        return op_list
