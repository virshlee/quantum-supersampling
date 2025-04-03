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



