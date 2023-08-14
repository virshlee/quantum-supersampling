import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, Rgate, MeasureHomodyne
from strawberryfields.tdm import shift_by
import numpy as np
from strawberryfields.tdm import reshape_samples

#this is a test algorithm implemented on strawberry fields

prog1 = sf.Program(2)
n = 20
r = 1.0
length = n - 1
alpha = [np.pi / 4] * length
phi = [0] * length
theta = [0] * length

N = 2  # Number of concurrent modes
prog2 = sf.TDMProgram(N=N)

with prog2.context(alpha, phi, theta) as (p, q):
    Sgate(r, 0) | q[1]
    BSgate(p[0]) | (q[0], q[1])
    Rgate(p[1]) | q[1]
    MeasureHomodyne(p[2]) | q[0]
eng2 = sf.Engine("gaussian")
result2 = eng2.run(prog2)
print(result2.samples)

