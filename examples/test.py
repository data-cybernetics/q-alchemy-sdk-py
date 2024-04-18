import os

import pennylane as qml
from pennylane import numpy as np

from q_alchemy.qiskit_to_pennylane import QAlchemyStatePreparation

os.environ["Q_ALCHEMY_API_KEY"] = "IUY65XTOKGwaFeuTxu48MnHR18UyVahZ"
dev = qml.device('default.qubit', wires=3)


@qml.qnode(dev)
def circuit(state):
    QAlchemyStatePreparation(state_vector=state, wires=range(3))
    return qml.state()


if __name__ == "__main__":

    state = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    state = state / np.linalg.norm(state)

    print(qml.draw(circuit, expansion_strategy="device", max_length=120)(state))