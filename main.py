from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mygates import *

shots = int(np.power(10, 7))
dev1 = qml.device("default.qubit", wires=8, shots=shots)


@qml.qnode(dev1, interface="autograd")
def grover4():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    Oracle(wires=[0, 1, 2, 3])
    Diffuser(wires=[0, 1, 2, 3])
    Oracle(wires=[0, 1, 2, 3])
    Diffuser(wires=[0, 1, 2, 3])
    Oracle(wires=[0, 1, 2, 3])
    Diffuser(wires=[0, 1, 2, 3])
    return qml.probs(wires=range(4))


@qml.qnode(dev1, interface="autograd")
def test4(params):
    qml.RX(phi=np.pi * params[0], wires=0)
    qml.RX(phi=np.pi * params[1], wires=1)
    qml.RX(phi=np.pi * params[2], wires=2)
    qml.RX(phi=np.pi * params[3], wires=3)
    CCCZ(wires=range(4))
    return qml.probs(wires=range(4))


@qml.qnode(dev1, interface="autograd")
def test():
    qml.QFT(wires=[4, 5, 6])
    qml.adjoint(qml.QFT(wires=[4, 5, 6]))
    return qml.probs(wires=range(4))


@qml.qnode(dev1, interface="autograd")
def grover7(inversion):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

    qml.Hadamard(wires=4)
    qml.Hadamard(wires=5)
    qml.Hadamard(wires=6)

    # qml.PauliX(wires=4)
    # qml.PauliX(wires=5)
    # qml.PauliX(wires=6)

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 4])
    Cdiffuser(wires=[0, 1, 2, 3, 4])

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 5])
    Cdiffuser(wires=[0, 1, 2, 3, 5])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 5])
    Cdiffuser(wires=[0, 1, 2, 3, 5])

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])

    qml.adjoint(qml.QFT(wires=[4, 5, 6]))
    # qml.Permute([0, 1, 2, 3, 6, 5, 4, 7], dev1.wires)
    # qml.Permute([1, 2, 3, 0, 5, 4, 6, 7], dev1.wires)
    qml.Permute([1, 2, 3, 0, 6, 5, 4, 7], dev1.wires)  # video
    # return qml.probs(wires=[4, 5, 6])
    return qml.probs(range(7))


@qml.qnode(dev1, interface="autograd")
def grover8(inversion):
    # qml.PauliX(wires=0)
    # qml.PauliX(wires=1)
    # qml.PauliX(wires=2)
    # qml.PauliX(wires=3)
    # qml.PauliX(wires=4)
    # qml.PauliX(wires=5)
    # qml.PauliX(wires=6)
    # qml.PauliX(wires=7)

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)
    qml.Hadamard(wires=5)
    qml.Hadamard(wires=6)
    qml.Hadamard(wires=7)

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 4])
    Cdiffuser(wires=[0, 1, 2, 3, 4])

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 5])
    Cdiffuser(wires=[0, 1, 2, 3, 5])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 5])
    Cdiffuser(wires=[0, 1, 2, 3, 5])

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 6])
    Cdiffuser(wires=[0, 1, 2, 3, 6])

    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])
    Coracle(inversion=inversion, wires=[0, 1, 2, 3, 7])
    Cdiffuser(wires=[0, 1, 2, 3, 7])

    #qml.adjoint(qml.QFT(wires=[4, 5, 6, 7]))

    qml.Permute([1, 2, 3, 0, 4, 5, 6, 7], dev1.wires)
    return qml.probs(wires=range(8))


def run8(inversion):
    probs = grover8(inversion = inversion)
    colormap = np.zeros((16, 16), dtype=float)
    i = 0
    j = 0
    index = 0
    for x in probs:
        colormap[i, j] = x
        print(index, x)
        index = index + 1
        i = (i + 1) % 16
        if i % 16 == 0:
            j = j + 1

    fig, ax = plt.subplots()

    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:03b}"))
    ax.yaxis.set_ticks(range(16))
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:04b}"))
    ax.xaxis.set_ticks(range(16))

    # plt.ylabel("")
    plt.xlabel("pixel index")
    plt.ylabel("grover iteration number")

    plt.imshow(colormap, cmap="Blues", interpolation='nearest')
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()
    # drawer = qml.draw(test)
    # print(drawer())


def run7(inversion):
    probs = grover7(inversion)

    colormap = np.zeros((8, 16), dtype=float)
    i = 0
    j = 0
    index = 0
    for x in probs:
        colormap[i, j] = x
        index = index + 1
        i = (i + 1) % 8
        if i % 8 == 0:
            j = j + 1

    fig, ax = plt.subplots()

    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:03b}"))
    ax.yaxis.set_ticks(range(16))
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:04b}"))
    ax.xaxis.set_ticks(range(16))

    # plt.ylabel("")
    plt.xlabel("pixel index")
    plt.ylabel("grover iteration number")

    plt.imshow(colormap, cmap="Blues", interpolation='nearest')
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    plt.show()
    # drawer = qml.draw(test)
    # print(drawer())


#run8(inversion = 6)
run7(inversion=6)
# print(grover7(inversion=5))
