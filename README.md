# 2D ME Resonator Shape Optimization

## Generating Sensible Shapes

Initially, the idea was to generate sensible shapes from a given set of
horizontal and vertical line lengths. However, combining these two is quite
complex because they do depend on eachother. For now, it seems therefore more
sensible to stick to only varying the horizontal lengths.

While separating the horizontal segments into discrete segments is vital to
define the number of free parameters, their values can, in principle, be
arbitrary. This probably makes the problem simpler rather than more complicated,
since sensible optimization algorithms can be applied.

It turns out though, that this problem (at least with low parameterization - a
large grid) can be solved quite easily with gradient descent in a short time.

## Solving Linear Elasticity (Fenics)

```python

```

## Determining the Relevant Mode

## Derivative Free Optimization Methods

### Gradient Descent - Direct Search

A nearest neighbour must be determined for each feature to determine a discrete
gradient. Starting from an initial shape this means that N resonance frequencies
must be calculated with N being the number of features. E.g. for 24 features and
an average calculation time of 12 s this means a total time of about 5 minutes.
However, with the right learning parameters quick convergence can be observed in
only about 30 iterations so that 2 hours of simulation time on a laptop are
reasonable to find a minimum.

For lower numbers of segements (up to 12 features) I found quick convergence in
what I think is the global optimium. However, when increasing the number of
segments (>=24) the algorithm gets stuck in a local minimum (I know it is a
local minimum because the resonance frequency is higher than for the 12 segment
case).

## Simulated Annealing

Source: de.wikipedia.org/wiki/Simulated_Annealing

Especially at low temperatures in many dimensions this algorithm performs quite
badly, since it is just a stochastic hill climbing algorithm. This is because random shapes do
only seldomly produce lower resonance frequencies when we are close to the
global maximum. The chance of producing a lower resonance frequency among the
many nearest neighbours is in the worst case 3$^N$ where n is the number of features.

It is therefore sensible to do a rough optimization with the simulated annealing

## Particle Swarm Algorithm

Sources:

- https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
- https://de.wikipedia.org/wiki/Partikelschwarmoptimierung
- https://web2.qatar.cmu.edu/~gdicaro/15382/additional/CompIntelligence-Engelbrecht-ch16.pdf

For convergence

$$w > 0.5 * (c1 + c2) − 1$$

where w is the inertia factor and c1 and c2 are the cognitive and social weights

- 10-30 particles should be selected
- c1 and c2 should be of similar size to balance local exploration and the interaction between the particles.

### Parameter tuning

inertia = 0.1
cognitive_weight = social_weight = 1.5
v_0 = 0

Very slow movement probably mostly governed by the randomness of the cognitive and social weights.

inertia = 0.2
cognitive_weight = social_weight = 1.0
v_0 = 0

No convergence even after 200 iterations for none of the particles. Very chaotic behaviour with a minimum of 105 kHz

inertia = 0.4
cognitive_weight = social_weight = 1.0
v_0 = 0

No convergence even after 200 iterations for none of the particles. Very chaotic behaviour with a minimum of 105 kHz

inertia = 0.8
cognitive_weight = social_weight = 1.0
v_0 = 0

No convergence even after 200 iterations for none of the particles. Chaotic behaviour with a minimum of 105 kHz

inertia = 1.0
cognitive_weight = social_weight = 1.0
v_0 = 0

Much better exploration with a minimum of 75 kHz, no conversion nevertheless.

inertia = 1.5
cognitive_weight = social_weight = 1.0
v_0 = 0

Quickly convering after about 40 iterations for particle 1. Both particles
converge to the same minimum after about 200 iterations. A minimum of 90000 was
found and it feels like there is some lack of exploration.

inertia = 2.0
cognitive_weight = social_weight = 1.0
v_0 = 0

Converges to something that is not the global minimum but takes longer than for
1.5 inertia. My feeling is that now the inertia is too high to allow for
directional changes. If they are present, they might then be too permanent in a
sense.

| inertia | cognitive weight | social weight | convergence | minimum resonance frequency | comments                                          |
| ------- | ---------------- | ------------- | ----------- | --------------------------- | ------------------------------------------------- |
| 0.1     | 1.5              | 1.5           | no          |                             |                                                   |
| 0.2     | 1.0              | 1.0           | no          | 105 kHz                     |                                                   |
| 0.4     | 1.0              | 1.0           | no          | 105 kHz                     |                                                   |
| 0.8     | 1.0              | 1.0           | no          | 105 kHz                     |                                                   |
| 1.0     | 1.0              | 1.0           | no          | 75 kHz                      |                                                   |
| 1.1     | 1.0              | 1.0           | no          | 74 kHz                      |                                                   |
| 1.2     | 1.0              | 1.0           | not really  | 86 kHz                      | not fully converging to minimum but getting close |
| 1.4     | 1.0              | 1.0           | yes         | 89 kHz                      |                                                   |
| 1.5     | 1.0              | 1.0           | yes         | 90 kHz                      |                                                   |
| 1.6     | 1.0              | 1.0           | no          | 70 kHz                      |                                                   |
| 1.7     | 1.0              | 1.0           | kind of     | 86 kHz                      |                                                   |
| 1.8     | 1.0              | 1.0           | yes         | 85 kHz                      |                                                   |
| 1.9     | 1.0              | 1.0           | kind of     | 88 kHz                      |                                                   |
| 2.0     | 1.0              | 1.0           | not really  | 85 kHz                      |                                                   |
| 1.4     | 0.5              | 0.5           | yes         | 90 kHz                      |                                                   |
| 1.4     | 0.5              | 1.0           | kind off    | 87 kHz                      |                                                   |
| 1.4     | 0.5              | 1.5           | yes, quick  | 95 kHz                      |                                                   |
| 1.4     | 0.5              | 2.0           |             | kHz                         |                                                   |
| 1.4     | 0.5              | 2.5           |             | kHz                         |                                                   |
| 1.4     | 1.0              | 0.5           |             | kHz                         |                                                   |
| 1.4     | 1.0              | 1.0           |             | kHz                         |                                                   |
| 1.4     | 1.0              | 1.5           |             | kHz                         |                                                   |
| 1.4     | 1.0              | 2.0           |             | kHz                         |                                                   |
| 1.4     | 1.0              | 2.5           |             | kHz                         |                                                   |
| 1.4     | 1.5              | 0.5           |             | kHz                         |                                                   |
| 1.4     | 1.5              | 1.0           |             | kHz                         |                                                   |
| 1.4     | 1.5              | 1.5           |             | kHz                         |                                                   |
| 1.4     | 1.5              | 2.0           |             | kHz                         |                                                   |
| 1.4     | 1.5              | 2.5           |             | kHz                         |                                                   |
| 1.4     | 2.0              | 0.5           |             | kHz                         |                                                   |
| 1.4     | 2.0              | 1.0           |             | kHz                         |                                                   |
| 1.4     | 2.0              | 1.5           |             | kHz                         |                                                   |
| 1.4     | 2.0              | 2.0           |             | kHz                         |                                                   |
| 1.4     | 2.0              | 2.5           |             | kHz                         |                                                   |
| 1.4     | 2.5              | 0.5           |             | kHz                         |                                                   |
| 1.4     | 2.5              | 1.0           |             | kHz                         |                                                   |
| 1.4     | 2.5              | 1.5           |             | kHz                         |                                                   |
| 1.4     | 2.5              | 2.0           |             | kHz                         |                                                   |
| 1.4     | 2.5              | 2.5           |             | kHz                         |                                                   |

- An inertia between 1.2 and 1.8 or so seems fine (also depends a bit on the starting point if it converges after 200 iterations or not). Further optimization might be necessary when changing the other two parameters but for now I choose an inertia of 1.4.

#### Conclusions

My feeling about it is that the closer you are to 0, the more chaotic turns particles make.

## Genetic Algorithms

Or memetic algorithm

## Deep Reinforcement Learning
