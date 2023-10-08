import math
from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
from scipy import sparse


class Uv(qml.operation.Operation):
    num_wires = 3
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.RX(np.pi, wires=wires[1]),
            qml.Toffoli(wires=wires),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]

class U0(qml.operation.Operation):
    num_wires = 3
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.RX(np.pi, wires=wires[1]),
            qml.Toffoli(wires=wires),
            qml.RX(np.pi, wires=wires[1]),

        ]

    def adjoint(self):
        return [
            qml.RX.adjoint(np.pi, wires=self.wires[1]),
            qml.Toffoli.adjoint(wires=self.wires),
            qml.RX.adjoint(np.pi, wires=self.wires[1]),
        ]

class Comparator3bit(qml.operation.Operation):
    num_wires = 9
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            Uv(wires=["a1", "b1", "target1"]),
            Uv(wires=["a2", "b2", "target2"]),
            Uv(wires=["a3", "b3", "target3"]),
            qml.Toffoli(wires=["b1", "target2", "target1"]),
            U0(wires=["a2", "b2", "target2"]),
            qml.Toffoli(wires=["b1", "b2", "target2"]),
            qml.Toffoli(wires=["target2", "target3", "target1"]),
            U0(wires=["a3", "b3", "target3"]),
            qml.Toffoli(wires=["target2", "b2", "target3"]),
        ]


class COMP(qml.operation.Operation):
    num_wires = 12
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            Uv(wires=["a1", "b1", "target1"]),
            Uv(wires=["a2", "b2", "target2"]),
            Uv(wires=["a3", "b3", "target3"]),
            Uv(wires=["a4", "b4", "target4"]),
            qml.Toffoli(wires=["b1", "target2", "target1"]),
            U0(wires=["a2", "b2", "target2"]),
            qml.Toffoli(wires=["b1", "b2", "target2"]),
            qml.Toffoli(wires=["target3", "target2", "target1"]),
            U0(wires=["a3", "b3", "target3"]),
            qml.Toffoli(wires=["target2", "b2", "target3"]),
            qml.Toffoli(wires=["target4", "target3", "target1"]),
        ]


class CCCZ(qml.operation.Operation):
    num_wires = 4
    num_params = 0

    def adjoint(self):
        return CCCZ(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    @property
    def control_wires(self):
        return qml.Wires(self.wires[:3])

    @property
    def is_hermitian(self):
        return True

    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )


class CCCCZ(qml.operation.Operation):
    num_wires = 5
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Z"

    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )

    def adjoint(self):
        return CCCCZ(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    @property
    def control_wires(self):
        return qml.Wires(self.wires[:3])

    @property
    def is_hermitian(self):
        return True

class Coracle(qml.operation.Operation):
    num_wires = 5
    num_params = 1
    """
    wires[4] is control wire
    the gate represents a controllable oracle that inverts inversion number of qubits when control qubit is inverted 
    """

    def __init__(self, inversion, wires, do_queue=True, id=None):

        super().__init__(inversion, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(inversion, wires):
        match inversion:
            case 0:
                return []
            case 1:
                return CCCCZ(wires=wires)
            case 6:
                return [
                    qml.CCZ(wires=[wires[0], wires[2], wires[4]]),
                    qml.CCZ(wires=[wires[1], wires[3], wires[4]]),
                    qml.RX(np.pi, wires=wires[1]),
                    qml.RX(np.pi, wires=wires[3]),
                    CCCZ(wires=[wires[1], wires[2], wires[3], wires[4]]),
                    qml.RX(np.pi, wires=wires[1]),
                    qml.RX(np.pi, wires=wires[3]),
                ]


class Cdiffuser(qml.operation.Operation):
    num_wires = 5
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.Hadamard(wires=wires[0]),
            qml.Hadamard(wires=wires[1]),
            qml.Hadamard(wires=wires[2]),
            qml.Hadamard(wires=wires[3]),
            qml.RX(np.pi, wires=wires[0]),
            qml.RX(np.pi, wires=wires[1]),
            qml.RX(np.pi, wires=wires[2]),
            qml.RX(np.pi, wires=wires[3]),
            CCCCZ(wires=wires),
            qml.RX(np.pi, wires=wires[0]),
            qml.RX(np.pi, wires=wires[1]),
            qml.RX(np.pi, wires=wires[2]),
            qml.RX(np.pi, wires=wires[3]),
            qml.Hadamard(wires=wires[0]),
            qml.Hadamard(wires=wires[1]),
            qml.Hadamard(wires=wires[2]),
            qml.Hadamard(wires=wires[3]),
        ]


class Oracle(qml.operation.Operation):
    num_wires = 4
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.CZ(wires=[0, 2]),
            qml.CZ(wires=[1, 3]),
            qml.RX(np.pi, wires=1),
            qml.RX(np.pi, wires=3),
            qml.CCZ(wires=[1, 2, 3]),
            qml.RX(np.pi, wires=1),
            qml.RX(np.pi, wires=3),
        ]


class Diffuser(qml.operation.Operation):
    num_wires = 4
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.Hadamard(wires=0),
            qml.Hadamard(wires=1),
            qml.Hadamard(wires=2),
            qml.Hadamard(wires=3),
            qml.RX(np.pi, wires=0),
            qml.RX(np.pi, wires=1),
            qml.RX(np.pi, wires=2),
            qml.RX(np.pi, wires=3),
            CCCZ(wires=[0, 1, 2, 3]),
            qml.RX(np.pi, wires=0),
            qml.RX(np.pi, wires=1),
            qml.RX(np.pi, wires=2),
            qml.RX(np.pi, wires=3),
            qml.Hadamard(wires=0),
            qml.Hadamard(wires=1),
            qml.Hadamard(wires=2),
            qml.Hadamard(wires=3),
        ]


class Of(qml.operation.Operation):
    """
    this gate implements the function
    mapping:
    |x>|0> --->|x>|f(x)>
    """
    num_wires = 4
    num_params = 0

    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        m = np.zeros((16, 16))
        for i in range(16):
            j = int(math.pow(i, 2) / 16)
            m[j, i] = 1
            np.linalg.det(m);
            return m


class COMP(qml.operation.Operation):
    """
    this gate implements the comparison between two states
    |a>|b>|0> ---> |a>|b>|a<b>
    """
    num_wires = 12
    num_params = 0

    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        m = np.zeros((16, 16))
        for i in range(16):
            j = int(math.pow(i, 2) / 16)
            m[j, i] = 1
            np.linalg.det(m);
        return m



