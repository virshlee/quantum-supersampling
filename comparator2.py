from functools import lru_cache

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mygates import *
from matplotlib.ticker import StrMethodFormatter

shots = int(np.power(10, 2))
dev1 = qml.device("default.qubit", wires=["a1","b1","target1","a2","b2","target2","a3","b3","target3"], shots=shots)



@qml.qnode(dev1, interface="autograd")
def compare_the_bits(a,b):
    qml.RX(np.pi * a[0], wires="a1"),
    qml.RX(np.pi * a[1], wires="a2"),
    qml.RX(np.pi * a[2], wires="a3"),
    qml.RX(np.pi * b[0], wires="b1"),
    qml.RX(np.pi * b[1], wires="b2"),
    qml.RX(np.pi * b[2], wires="b3"),
    Comparator3bit(wires=["a1","b1","target1","a2", "b2", "target2","a3", "b3", "target3"])
    return qml.probs(wires=["target1"])


for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                for i5 in range(2):
                    for i6 in range(2):
                        a = compare_the_bits(a=[i1, i2, i3], b=[i4, i5, i6])
                        print("a = ", [i1, i2, i3], "b = ", [i4, i5, i6], "a<=b |a > b: ", a)
