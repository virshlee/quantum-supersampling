# quantum-supersampling

This is a quantum supersampler implemented with penny lane.
The main purpose of a supersampler is to calculate the average (or integral) of an f function over a set using less samples than the number of elements in the set.
In classical numerical computing this can be done with monte carlo integration.

The circuit was implemented from the paper
https://history.siggraph.org/learning/quantum-supersampling/
and a brief video overview
https://vimeo.com/180284417

The motivation came from https://arxiv.org/abs/2211.03418
where the algorithm was used on f(j) function, where f(j) represents the ray energy (real value) of the j-th ray.

The implementation on a real quantum computer leads to a problem that will likely appear in every quantum algorithm:

Decomposition:
   The quantum circuit must be implemented with the quantum gates that are available on the target hardware.
   This problem leads to decomposition. I found this paper, it seems to be useful.
https://arxiv.org/abs/quant-ph/9503016

Another challenge will probably come up if we use photonic quantum computers:
The default quantum gates used in quantum circuits are besd on qubits, however the state of photons cannot be described by qubits but with qumodes.  
We have to implement the gates such as hadamard, cnot with gates that are available on a photonic computer e.g. squeezer, beam-splitter. This can be done in several ways, in Xanadu they say the best method is made with GKP states
https://pennylane.ai/qml/demos/tutorial_photonics


