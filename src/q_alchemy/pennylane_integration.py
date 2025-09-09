# Copyright 2022-2023 data cybernetics ssc GmbH.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

from scipy.sparse import csr_matrix

import pennylane as qml
import numpy as np
from pennylane import math
from pennylane.ops.qubit.state_preparation import StatePrep
from pennylane.operation import Operation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from q_alchemy.initialize import q_alchemy_as_qasm, OptParams

# Normalization precision required for compatibility with Qiskit and qclib state preparation.
ATOL = 1e-10
RTOL = 1e-9

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

    @staticmethod
    def _preprocess(state, wires, pad_with, normalize, validate_norm):
        """Validate and pre-process inputs as follows:

        * If state is batched, the processing that follows is applied to each state set in the batch.
        * Check that the state tensor is one-dimensional.
        * If pad_with is None, check that the last dimension of the state tensor
          has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
          last dimension of the state tensor is not larger than :math:`2^n` and pad state
          with value if necessary.
        * If normalize is false, check that last dimension of state is normalised to one. Else, normalise the
          state tensor.
        """
        if isinstance(state, (list, tuple)):
            state = math.array(state)

        if "torch" in str(state.dtype):
            state = state.detach().numpy()

        # Promote from `float32` to `float64` to ensure normalization meets the required precision.
        if "float32" in str(state.dtype):
            state = state.astype(np.float64, copy=False)
        if "complex64" in str(state.dtype):
            state = state.astype(np.complex128, copy=False)

        shape = math.shape(state)

        # check shape
        if len(shape) not in (1, 2):
            raise ValueError(
                f"State must be a one-dimensional tensor, or two-dimensional with batching; got shape {shape}."
            )

        n_states = shape[-1]
        dim = 2 ** len(Wires(wires))
        if pad_with is None and n_states != dim:
            raise ValueError(
                f"State must be of length {dim}; got length {n_states}. "
                f"Use the 'pad_with' argument for automated padding."
            )

        if pad_with is not None:
            normalize = True
            if n_states > dim:
                raise ValueError(
                    f"Input state must be of length {dim} or "
                    f"smaller to be padded; got length {n_states}."
                )

            # pad
            if n_states < dim:
                padding = [pad_with] * (dim - n_states)
                if len(shape) > 1:
                    padding = [padding] * shape[0]
                padding = math.convert_like(padding, state)
                state = math.hstack([state, padding])

        if not validate_norm:
            return state

        # normalize
        if "int" in str(state.dtype):
            state = math.cast_like(state, 0.0)

        norm = math.linalg.norm(state, axis=-1)

        if math.is_abstract(norm):
            if normalize:
                state = state / math.reshape(norm, (*shape[:-1], 1))

        elif not math.allclose(norm, 1.0, atol=ATOL, rtol=RTOL):
            if normalize:
                state = state / math.reshape(norm, (*shape[:-1], 1))
            else:
                raise ValueError(
                    f"The state must be a vector of norm 1.0; got norm {norm}. "
                    "Use 'normalize=True' to automatically normalize."
                )

        return state

class QAlchemyStatePreparation(Operation):
    def __init__(self, state_vector, wires, id=None, **kwargs):

        # Right now, only the basis gates as given below can be set.
        if "opt_params" in kwargs:
            opt_params = kwargs["opt_params"]
        else:
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
                if not qml.math.allclose(norm, 1.0, atol=ATOL, rtol=RTOL):
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

        qasm, summary = q_alchemy_as_qasm(state_vector, opt_params, return_summary=True)
        loaded_circuit = qml.from_qasm(qasm)
        # Reorder the wires, as the original qasm code assumes qubit 0 is the least significant bit.
        qs = qml.tape.make_qscript(loaded_circuit)(wires=wires[::-1])
        # Add explicit globalphase to the start.
        return [qml.GlobalPhase(-summary["global_phase"])] + qs.operations
