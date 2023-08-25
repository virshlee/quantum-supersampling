from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mygates import *
from matplotlib.ticker import StrMethodFormatter

shots = int(np.power(10, 7))
dev1 = qml.device("default.qubit", wires=[0, 1, 2, 3], shots=shots)

class Oracle4(qml.operation.Operation):
    num_wires = 4
    num_params = 0

    @staticmethod
    def compute_decomposition(wires):
        return [
            qml.CZ(wires=[0, 2]),
            qml.CZ(wires=[1, 3]),
            qml.PauliX(wires=1),
            qml.PauliX(wires=3),
            qml.CCZ(wires=[1, 2, 3]),
            qml.PauliX(wires=1),
            qml.PauliX(wires=3),
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
    Oracle4(wires=[0, 1, 2, 3])
    Diffuser0(wires=[0, 1, 2, 3])
    Oracle4(wires=[0, 1, 2, 3])
    Diffuser0(wires=[0, 1, 2, 3])
    qml.Permute([3, 0, 1, 2], dev1.wires)
    return qml.probs(wires=range(4))




a = grover4()
print(a)
c = np.zeros((4, 4), dtype=float)
i=0
j=0
index=0
for x in a:
    c[i, j] = x
    print(index, x)
    index=index+1
    i = (i + 1) % 4
    if i % 4 == 0:
        j = j + 1


print(c)

fig, ax = plt.subplots()

#ax.yaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.yaxis.set_ticks(range(4))
#ax.xaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.xaxis.set_ticks(range(4))

plt.xlabel("qx")
plt.ylabel("qy")
plt.imshow(c, cmap='hot', interpolation='nearest')
plt.show()