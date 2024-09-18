import logging

import pennylane as qml
from pennylane.operation import Operation

from q_alchemy.initialize import q_alchemy_as_qasm, OptParams
from q_alchemy.parser.qasm_pennylane import from_qasm

LOG = logging.getLogger(__name__)


class QAlchemyStatePreparation(Operation):
    def __init__(self, state_vector, wires, id=None, **kwargs):

        # Right now, only the basis gates as given below can be set.
        if "opt_params" in kwargs:
            opt_params = kwargs["opt_params"]
        else:
            if "basis_gates" in kwargs:
                raise Warning(f"Basis Gates cannot be set currently. The inpot will be ignored.")
            opt_params = OptParams.from_dict(kwargs)

        # Append options
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

    # noinspection PyMethodOverriding
    @staticmethod
    def compute_decomposition(state_vector, wires, **hyperparameters):  # pylint: disable=arguments-differ
        opt_params = hyperparameters.get("opt_params", OptParams(basis_gates=["id", "rx", "ry", "rz", "cx"]))
        if len(qml.math.shape(state_vector)) > 1:
            raise ValueError(
                "Broadcasting with QAlchemyStatePreparation is not supported. Please use the "
                "qml.transforms.broadcast_expand transform to use broadcasting with "
                "QAlchemyStatePreparation."
            )

        qasm = q_alchemy_as_qasm(state_vector, opt_params)
        return from_qasm(qasm)
