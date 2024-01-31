import numpy as np
import time
from qulacs.gate import CZ

from numpy.random import *

from numpy.typing import NDArray
from multiprocessing import cpu_count
from functools import reduce
import operator
from skqulacs.circuit import LearningCircuit
import time

def seconds2hms(seconds):
    days = int(seconds // 86400)
    time_struct = time.gmtime(seconds)
    return f"{days} d: {time_struct.tm_hour} h: {time_struct.tm_min} m: {time_struct.tm_sec} s:"


def total_grid_estimation(search_dict, n_splits, single_time=500, task_speed=0.02, num_process=None):
    if num_process is None:
        num_process = cpu_count() - 1
    total_combinations = reduce(operator.mul, (len(v) for v in search_dict.values()), 1)
    
    total_rounds = total_combinations*n_splits / num_process
    print(f"Total number of grid params: {total_combinations}, total num of combination of CV,grid: {total_combinations*n_splits}")
    print(f'total rounds of parallel processes {total_rounds}')
    print(f'Estimated time: {seconds2hms(single_time * np.ceil(total_rounds))}')
    print(f'Estimated time by speed: {seconds2hms(total_combinations / task_speed)}')
    
    

def create_jerbi_ansatz(n_qubit, train_depth, seed=0):
    """create circuit used in https://www.nature.com/articles/s41467-023-36159-y.
    ibm circuit的なembeddingのあとにHEA
    ibm circuitとは少し違って、ZZ rotationのパラメータが、x_i x_j で、j<i すべてについてとっている。
    entangling layerは、circular boundary condition
    Args:
        n_qubits: number of qubits
        train_depth: number of layers of trainable parameters
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa: float = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        #j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        for j in range(i):
            circuit.add_CNOT_gate(i, j)
            circuit.add_input_RZ_gate(
                j,
                lambda x, i=i: (
                    preprocess_x(x, i) * preprocess_x(x, j)
                ),
            )
            circuit.add_CNOT_gate(i, j)

    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        #j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        for j in range(i):
            circuit.add_CNOT_gate(i, j)
            circuit.add_input_RZ_gate(
                j,
                lambda x, i=i: (
                    preprocess_x(x, i) * preprocess_x(x, j)
                ),
            )
            circuit.add_CNOT_gate(i, j)
            
    rng = default_rng(seed)
    for k in range(train_depth):
        for i in range(n_qubit):
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
        for i in range(n_qubit):
            j = (i + 1) % n_qubit
            circuit.add_gate(CZ(i, j))
    for i in range(n_qubit):
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RX_gate(i, angle)
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RY_gate(i, angle)
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RZ_gate(i, angle)
    return circuit

def create_ibm_HEA_ansatz(n_qubit, train_depth, seed=0):
    """create circuit used in https://www.nature.com/articles/s41467-023-36159-y.
    ibm circuitのembeddingのあとにHEA
    entangling layerは、circular boundary condition
    Args:
        n_qubits: number of qubits
        train_depth: number of layers of trainable parameters
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa: float = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j,
            lambda x, i=i: (
                (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
            ),
        )
        circuit.add_CNOT_gate(i, j)

    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j,
            lambda x, i=i: (
                (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
            ),
        )
        circuit.add_CNOT_gate(i, j)
            
    rng = default_rng(seed)
    for k in range(train_depth):
        for i in range(n_qubit):
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
        for i in range(n_qubit):
            j = (i + 1) % n_qubit
            circuit.add_gate(CZ(i, j))
    for i in range(n_qubit):
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RX_gate(i, angle)
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RY_gate(i, angle)
        angle = 2.0 * np.pi * rng.random()
        circuit.add_parametric_RZ_gate(i, angle)
    return circuit



