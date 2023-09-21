import matplotlib.pyplot as plt
from mygates import *

shots = int(np.power(10, 7))
dev1 = qml.device("default.qubit", wires=[0, 1, 2, 3], shots=shots)

s = 0

for i in range(1, 30):
    s += i

weights= np.arange(30)/s

#print(weights)

#print(qml.ArbitraryStatePreparation(weights, wires=[0, 1, 2, 3]).compute_decomposition(weights, wires=[0, 1, 2, 3]))

@qml.qnode(dev1, interface="autograd")
def prepa():
    qml.ArbitraryStatePreparation(weights, wires=[0, 1, 2, 3])


    return qml.probs(wires=range(4))

print(qml.draw(prepa))
a = prepa()
c = np.zeros((4, 4), dtype=float)
i=0
j=0
index=0
for x in a:
    c[i, j] = x
    #print(index, x)
    index=index+1
    i = (i + 1) % 4
    if i % 4 == 0:
        j = j + 1


#print(c)

fig, ax = plt.subplots()

#ax.yaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.yaxis.set_ticks(range(4))
#ax.xaxis.set_major_formatter(StrMethodFormatter("{x:02b}"))
ax.xaxis.set_ticks(range(4))

plt.xlabel("qx")
plt.ylabel("qy")
plt.imshow(c, cmap='hot', interpolation='nearest')
plt.show()

