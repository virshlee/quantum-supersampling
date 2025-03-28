from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mygates import *
from matplotlib.ticker import StrMethodFormatter

shots = int(np.power(10, 7))
dev1 = qml.device("default.qubit", wires=[0, 1, 2, 3, 4], shots=shots)


class Oracle4(qml.operation.Operation):
    num_wires = 4
    num_params = 0
    """
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
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
    """
    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.IntegerComparator(value=5,geq=False,wires=range(5)),
            qml.CZ(wires=[4, 0]),
            qml.CZ(wires=[4, 1]),
            qml.CZ(wires=[4, 2]),
            qml.CZ(wires=[4, 3]),
        ]


class Diffuser0(qml.operation.Operation):
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


@qml.qnode(dev1, interface="autograd")
def grover4():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    Oracle4(wires=[0, 1, 2, 3])
    Diffuser0(wires=[0, 1, 2, 3])
    return qml.probs(wires=range(4))


a = grover4()
print(a)
c = np.zeros((4, 4), dtype=float)
i = 0
j = 0
index = 0
for x in a:
    c[i, j] = x
    print(index, x)
    index = index + 1
    i = (i + 1) % 4
    if i % 4 == 0:
        j = j + 1

print(c)

fig, ax = plt.subplots()

# ax.yaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.yaxis.set_ticks(range(4))
# ax.xaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.xaxis.set_ticks(range(4))

plt.xlabel("qx")
plt.ylabel("qy")
plt.imshow(c, cmap='hot', interpolation='nearest')
plt.show()