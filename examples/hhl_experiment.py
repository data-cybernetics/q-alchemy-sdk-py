from qiskit import transpile
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from scipy.sparse import diags
import logging
from typing import Optional, Union, List, Callable, Tuple
from linear_solvers import NumPyLinearSolver, HHL
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Statevector
import sys
import os
sys.path.append('./')
from q_alchemy.qiskit import QAlchemyInitialize
import multiprocessing
import pandas as pd

data_path = 'examples/hhl_data'

def get_solution_vector(solution, length):
    raw_solution_vector = Statevector(solution.state)
    numq = solution.state.num_qubits
    n = 2**(numq-1)
    print(raw_solution_vector)
    solution_vector = Statevector(solution.state).data[n:n+length].real
    print(solution_vector)
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

def solve_HHL(A, b, length, fid_loss,type):
    if type == 'qalchemy':
        print('constructing circuit....')
        sp_org = QAlchemyInitialize(b, opt_params={f'max_fidelity_loss':fid_loss})
        b_qc = transpile(sp_org.definition, basis_gates=["id", "rx", "ry", "rz", "cx"])
        print(b_qc)
        print('solving....')
        hhl_solution = HHL().solve(A, b_qc)
        
    elif type == 'ibm':
        hhl_solution = HHL().solve(A, b)
        
    else:
        print('ERROR!')
        
    print('calculating results....')
        
    hhl_circuit = hhl_solution.state
    print(hhl_circuit)
    hhl_ans = get_solution_vector(hhl_solution, length)
    print(hhl_ans)
    hhl_norm = hhl_solution.euclidean_norm
    print(hhl_norm)
    
    return hhl_circuit, hhl_ans, hhl_norm

os.environ["Q_ALCHEMY_API_KEY"] = "JnvkpMCsyr4nB9nHcwa6CbxqhtZXyF1b"

logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=="__main__":
   
    A = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [3.0, 4.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 8.0, 9.0, 0.0, 10.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0]])
    
    b = np.random.rand(8)
    LOG.info(b)
    b = b/np.linalg.norm(b)
    step_size = 0.01
    np.savetxt(f'{data_path}/trial_solution_{step_size}.npy', b)
    b = np.loadtxt('examples/hhl_data/trial_solution.npy')
    print(b)
    fid_losses = np.linspace(0, 1, int(1/step_size)+1)
    print(fid_losses)
    
    length = 8
    
    ans = []
    norms = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []

        for fid_loss in fid_losses:
            print(f'caluating for {fid_loss}')
            result = pool.apply_async(solve_HHL, (A, b, length, fid_loss, 'qalchemy'))
            print('appending results')
            results.append(result)

            hhl_circuit, hhl_ans, hhl_norm = result.get()
            print(f"fid_loss: {fid_loss}, hhl_norm: {hhl_norm}")
            print(hhl_circuit)
            ans.append([hhl_ans])
            norms.append(hhl_norm)
            
            final_results = { "solution": ans, "norms": norms}
            df = pd.DataFrame(final_results)
            df.to_csv(f'{data_path}/hhl_qalchemy_results_{step_size}.csv')
    
        final_results = { "fid_loss": fid_losses, "solution": ans, "norms": norms}
        df = pd.DataFrame(final_results)
        df.to_csv(f'{data_path}/hhl_qalchemy_results_{step_size}.csv')
    # df1 = pd.read_csv('examples/hhl_data/hhl_qalchemy_results_0.01_0.73.csv', index_col=0)
    # df2 = pd.read_csv('examples/hhl_data/hhl_qalchemy_results_0.74_0.01.csv', index_col=0)
    
    # df = pd.concat([df1, df2], ignore_index=True)
    # df.to_csv('examples/hhl_data/hhl_qalchemy_results_0_1.00.csv')

    
    # for fidel_loss in fidel_losses:
    #     LOG.info(f'running for {fidel_loss}')
    #     sp_org = QAlchemyInitialize(b, opt_params={f'max_fidelity_loss':fidel_loss})
    #     b_qc = transpile(sp_org.definition, basis_gates=["id", "rx", "ry", "rz", "cx"])
    #     x0 = 2**(14)
    #     circuit, sol, norm = solve_HHL(A, b_qc, x0, 8)
        
    #     circuits.append(circuit)
    #     solutions.append(sol)
    #     norms.append(norm)
        
        
        
    
    
