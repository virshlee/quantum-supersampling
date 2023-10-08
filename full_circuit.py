from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mygates import *
from matplotlib.ticker import StrMethodFormatter

shots = int(np.power(10, 2))
dev1 = qml.device("default.qubit", wires=["a1", "a2", "a3", "a4"], shots=shots)



"""
@qml.qnode(dev1, interface="autograd")
def circuit(a):
    qml.RX(np.pi * a[0], wires="a1"),
    qml.RX(np.pi * a[1], wires="a2"),
    qml.RX(np.pi * a[2], wires="a3"),
    qml.RX(np.pi * a[3], wires="a4"),
    #Of(wires=["a1", "a2", "a3", "a4"])
    return qml.probs(wires=["a1", "a2", "a3", "a4"])


for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                result = circuit(a=[i1, i2, i3, i4])
                print("x = ", [i1, i2, i3, i4], "y = ", result)

"""
dev2 = qml.device("default.qubit", wires=["a1", "b1", "target1", "a2", "b2", "target2", "a3", "b3", "target3", "a4",
                                          "b4", "target4"], shots=shots)


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
            qml.Toffoli(wires=["target2", "b3", "target3"]),
            qml.Toffoli(wires=["target4", "target3", "target1"]),
        ]

    def adjoint(self):
        return [
            qml.Toffoli.adjoint(wires=["target4", "target3", "target1"]),
            qml.Toffoli.adjoint(wires=["target2", "b3", "target3"]),
            U0.adjoint(wires=["a3", "b3", "target3"]),
            qml.Toffoli.adjoint(wires=["target3", "target2", "target1"]),
            qml.Toffoli.adjoint(wires=["b1", "b2", "target2"]),
            U0.adjoint(wires=["a2", "b2", "target2"]),
            qml.Toffoli.adjoint(wires=["b1", "target2", "target1"]),
            Uv.adjoint(wires=["a1", "b1", "target1"]),
            Uv.adjoint(wires=["a2", "b2", "target2"]),
            Uv.adjoint(wires=["a3", "b3", "target3"]),
            Uv.adjoint(wires=["a4", "b4", "target4"]),
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
            qml.Toffoli(wires=["target2", "b3", "target3"]),
            qml.Toffoli(wires=["target4", "target3", "target1"]),
        ]

    def adjoint(self):
        return [
            qml.Toffoli.adjoint(wires=["target4", "target3", "target1"]),
            qml.Toffoli.adjoint(wires=["target2", "b3", "target3"]),
            U0.adjoint(wires=["a3", "b3", "target3"]),
            qml.Toffoli.adjoint(wires=["target3", "target2", "target1"]),
            qml.Toffoli.adjoint(wires=["b1", "b2", "target2"]),
            U0.adjoint(wires=["a2", "b2", "target2"]),
            qml.Toffoli.adjoint(wires=["b1", "target2", "target1"]),
            Uv.adjoint(wires=["a1", "b1", "target1"]),
            Uv.adjoint(wires=["a2", "b2", "target2"]),
            Uv.adjoint(wires=["a3", "b3", "target3"]),
            Uv.adjoint(wires=["a4", "b4", "target4"]),
        ]


@qml.qnode(dev2, interface="autograd")
def compare_the_bits(a, b):
    qml.RX(np.pi * a[0], wires="a1"),
    qml.RX(np.pi * a[1], wires="a2"),
    qml.RX(np.pi * a[2], wires="a3"),
    qml.RX(np.pi * a[3], wires="a4"),
    qml.RX(np.pi * b[0], wires="b1"),
    qml.RX(np.pi * b[1], wires="b2"),
    qml.RX(np.pi * b[2], wires="b3"),
    qml.RX(np.pi * b[3], wires="b4"),
    COMP(wires=["a1", "b1", "target1", "a2", "b2", "target2", "a3", "b3", "target3", "a4", "b4", "target4"])
    return qml.probs(wires=["target1"])


for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                for i5 in range(2):
                    for i6 in range(2):
                        for i7 in range(2):
                            for i8 in range(2):
                                result = compare_the_bits(a=[i1, i2, i3, i4], b=[i5, i6, i7, i8])
                                print("a = ", [i1, i2, i3, i4], "b = ", [i5, i6, i7, i8], "a > b: ", result)




