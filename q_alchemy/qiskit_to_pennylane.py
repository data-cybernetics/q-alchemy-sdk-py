import pennylane as qml
from pennylane import numpy as np
import logging
import re
import qiskit
from qiskit import QuantumCircuit

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

class convert_qiskit:
    def __init__(self, prep_circuit: QuantumCircuit):
        #LOG.info('hi')
        self.prep_circuit = prep_circuit
        self.inst = prep_circuit.data
        self.num_qubits = int(prep_circuit.num_qubits)
        
    def extract_gates(self, inst_str:str):
        name_match = re.findall(r'name=\'(\w+)\'', inst_str)
        params_match = re.findall(r'params=\[(.*?)\]', inst_str)
        return name_match, params_match
    
    def extract_qubits(self, inst_str:str):
        qubit_matches = re.findall(r'\),\s(\d)', inst_str)
        if qubit_matches:
            return qubit_matches
        else:
            return None
    
    def circuit_to_list(self):
        n = len(self.inst)
        #LOG.info(n)
        gates = []
        params = []
        qubits = []
        for i in range(n):
            gate, param = self.extract_gates(str(self.inst[i][0]))
            qubit = self.extract_qubits(str(self.inst[i][1]))
            #print(f'adding gate {gate[0]}')
            gates.append(gate[0])
            params.append(param[0])
            qubits.append(qubit)
        return gates, params, qubits
    
    def list_to_pennylane(self, mode:str):
        
        gates, params, qubits = self.circuit_to_list()
        #LOG.info(len(gates))
        
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def circuit(gates, params, qubits):
            for i in range(len(gates)):
                if gates[i] == 'rz':
                    qml.RZ(float(params[i]), wires= self.num_qubits-1-int(qubits[i][0]))
                elif gates[i] == 'cx':
                    qml.CNOT(wires=[self.num_qubits-1-int(qubits[i][0]), self.num_qubits-1-int(qubits[i][1])])
                elif gates[i] == 'ry':
                    qml.RY(float(params[i]), wires=self.num_qubits-1-int(qubits[i][0]))
                elif gates[i] == 'rx':
                    qml.RY(float(params[i]), wires=self.num_qubits-1-int(qubits[i][0]))
                
            return qml.state()
        
        drawer = qml.draw(circuit)
        if mode == 'draw':
            print(drawer(gates, params, qubits))
        
        penny_state = circuit(gates, params, qubits)
        return gates, params, qubits, penny_state