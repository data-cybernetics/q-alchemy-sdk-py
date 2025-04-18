from typing import Optional, Union

import logging

from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.ops.qubit.state_preparation import StatePrep
from pennylane.operation import Operation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from q_alchemy.initialize import q_alchemy_as_qasm, OptParams

LOG = logging.getLogger(__name__)

class AmplitudeEmbedding(StatePrep):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        state: Union[TensorLike, csr_matrix],
        wires: WiresLike,
        pad_with=None,
        normalize=False,
        id: Optional[str] = None,
        validate_norm: bool = True,
        **kwargs
    ):
        self.kwargs = kwargs

        super().__init__(
            state,
            wires,
            pad_with,
            normalize,
            id,
            validate_norm
        )

        if "opt_params" in kwargs:
            opt_params = kwargs["opt_params"]
        else:
            opt_params = OptParams.from_dict(kwargs)

        self._hyperparameters['opt_params'] = opt_params

    # pylint: disable=unused-argument
    @staticmethod
    def compute_decomposition(state: TensorLike, wires: WiresLike, **kwargs) -> list[Operator]:
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.AmplitudeEmbedding.decomposition`.

        Args:
            state (array[complex]): a state vector of size 2**len(wires)
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.AmplitudeEmbedding.compute_decomposition(np.array([1, 0, 0, 0]), wires=range(2))
        [QAlchemyStatePreparation(tensor([1, 0, 0, 0], requires_grad=True), wires=[0, 1])]

        """

        return [QAlchemyStatePreparation(state, wires, id=None, **kwargs)]

class QAlchemyStatePreparation(Operation):
    def __init__(self, state_vector, wires, id=None, **kwargs):

        # Right now, only the basis gates as given below can be set.
        if "opt_params" in kwargs:
            opt_params = kwargs["opt_params"]
        else:
            if "basis_gates" in kwargs:
                raise Warning("Basis Gates cannot be set currently. The input will be ignored.")
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
        loaded_circuit = qml.from_qasm(qasm)
        # Reorder the wires, as the original qasm code assumes qubit 0 is the least significant bit.
        qs = qml.tape.make_qscript(loaded_circuit)(wires=wires[::-1])
        return qs.operations
