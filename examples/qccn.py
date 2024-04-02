import pennylane as qml
import numpy as np
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from sklearn.model_selection import train_test_split

from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
import os
import sys
from qiskit import transpile
import tensorflow as tf
import numpy as np
import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as pnp
import logging
from sklearn import datasets, metrics, svm

sys.path.append('./src')
from q_alchemy.qiskit_integration import QAlchemyInitialize, OptParams
import q_alchemy.qiskit_to_pennylane as qml_convert

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

os.environ["Q_ALCHEMY_API_KEY"] = "016UeTrQ4G6qcKIZ6vmLAhOnmLbQGd20"

#algorithm_globals.random_seed = 12345

def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def preparation_circuit(X):
    sp_org = QAlchemyInitialize(X, opt_params={'max_fidelity_loss':0.0})
    qc = transpile(sp_org.definition, basis_gates=["id", "rx", "ry", "rz", "cx"])
    return qc

dev = qml.device('default.qubit', wires = 6)
@qml.qnode(dev)
def QCNN(X, U, params, state_prep):
    if state_prep == 'q-alchemy':
        qml_convert.convert_qiskit(X, range(6))
    elif state_prep == 'motten':
         qml.MottonenStatePreparation(X, wires=range(6))
        

    U(params[:15], wires=[0, 5])
    for i in range(0, 6, 2):
        U(params[:15], wires=[i, i + 1])
    for i in range(1, 5, 2):
        U(params[:15], wires=[i, i + 1])
        
    U(params[15:30], wires=[0, 4])
    U(params[15:30], wires=[0, 2])
    U(params[15:30], wires=[2, 4])
    
    U(params[30:45], wires=[0,4])
    U(params[30:45], wires=[0,2])
    
    result = qml.probs(wires=0)
    return result

def cross_entropy(labels, predictions):
    loss = 0
    LOG.info(f'labels: {labels}')
    LOG.info(f'predictions:{predictions}')
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    LOG.info(f"loss: {loss}")
    return -1 * loss

def cost(v, X, Y, U, params, state_prep):
    pred = [QCNN(x, U, params, state_prep) for x in X]
    LOG.info(f"predictions: {pred}")
    loss = cross_entropy(Y, pred)
    return loss
    
def train_circuit(X_train, y_train, U, U_params, state_prep):
    params = pnp.random.randn(3*U_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=0.01)
    loss_history = []
    steps = 200
    batch_size = 25
    for it in range(steps):
        print("iteration: ", it)
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, params, state_prep),
                                                     params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
            np.save('loss.npy', loss_history)
            np.save('params.npy', params)
    return loss_history, params

def load_sklearn():
    classes = [0, 1]
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    y = digits.target
    
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.4, random_state=12345)
    
    x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
    x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

    X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
    Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]
    X_train_norms, X_test_norms = np.linalg.norm(X_train, axis=1), np.linalg.norm(X_test, axis=1)
    X_train_norms, X_test_norms = X_train_norms[:, np.newaxis], X_test_norms[:, np.newaxis]
    
    X_train, X_test = X_train/X_train_norms, X_test/X_test_norms
    
    return X_train, X_test, Y_train, Y_test

if __name__=='__main__':
    X_train, X_test, Y_train, Y_test = load_sklearn()
    X_train = X_train*1j
    print(X_train[0])
    
    loss_history, params = train_circuit(X_train, Y_train, U_SU4, 15, 'q-alchemy')
    # loss = np.load('loss.npy')
    # params = np.load('params.npy')
    # LOG.info(loss)
    # LOG.info(params)