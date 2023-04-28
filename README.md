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

$w > 0.5 * (c1 + c2) − 1$

where w is the inertia factor and c1 and c2 are the cognitive and social weights

- 10-30 particles should be selected
- c1 and c2 should be of similar size to balance local exploration and the interaction between the particles.

### Parameter tuning

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
| 1.4     | 0.5              | 1.0           | kind of     | 87 kHz                      |                                                   |
| 1.4     | 0.5              | 1.5           | yes, quick  | 95 kHz                      |                                                   |
| 1.4     | 0.5              | 2.0           | yes, quick  | 86 kHz                      |                                                   |
| 1.4     | 0.5              | 2.5           | no          | 92 kHz                      |                                                   |
| 1.4     | 1.0              | 0.5           | kind of     | 87 kHz                      |                                                   |
| 1.4     | 1.0              | 1.0           | no          | 81 kHz                      |                                                   |
| 1.4     | 1.0              | 1.5           | yes         | 85 kHz                      |                                                   |
| 1.4     | 1.0              | 2.0           | yes         | 84 kHz                      |                                                   |
| 1.4     | 1.0              | 2.5           | yes         | 86 kHz                      |                                                   |
| 1.4     | 1.5              | 0.5           | no          | 55 kHz                      |                                                   |
| 1.4     | 1.5              | 1.0           | no          | 85 kHz                      |                                                   |
| 1.4     | 1.5              | 1.5           | no          | 48 kHz                      |                                                   |
| 1.4     | 1.5              | 2.0           | kind of     | 60 kHz                      |                                                   |
| 1.4     | 1.5              | 2.5           | kind of     | 87 kHz                      |                                                   |
| 1.4     | 2.0              | 0.5           | no          | 93 kHz                      |                                                   |
| 1.4     | 2.0              | 1.0           | no          | 91 kHz                      |                                                   |
| 1.4     | 2.0              | 1.5           | no          | 86 kHz                      |                                                   |
| 1.4     | 2.0              | 2.0           | no          | 57 kHz                      |                                                   |
| 1.4     | 2.0              | 2.5           | kind of     | 70 kHz                      |                                                   |
| 1.4     | 2.5              | 0.5           | no          | 54 kHz                      |                                                   |
| 1.4     | 2.5              | 1.0           | no          | 71 kHz                      |                                                   |
| 1.4     | 2.5              | 1.5           | no          | 66 kHz                      |                                                   |
| 1.4     | 2.5              | 2.0           | no          | 82 kHz                      |                                                   |
| 1.4     | 2.5              | 2.5           | no          | 92 kHz                      |                                                   |

- An inertia between 1.2 and 1.8 or so seems fine (also depends a bit on the starting point if it converges after 200 iterations or not). Further optimization might be necessary when changing the other two parameters but for now I choose an inertia of 1.4.
- A higher social weight leads to a tendency of less exploration but quicker convergence. Too quick of a convergence (for too high ratios of social/cognitive weight) can therefore be detrimental as well.
- In principle all three parameters are interdependent so that convergence is not necessarily observed for a set of parameters if only one parameter is changed.

The grid search that I performed to optimize the parameters is not necessarily
ideal but has lead to a good starting point. Increasing the number of particles
will now hold interesting behaviour. So the probable ranges for the parameters
are:

1. Inertia: [1.2, 1.8]
2. Cognitive weight: [0.5, 1.0]
3. Social weight: [1.0, 2.5]

The ideal solution is probably somewhere at inertia: 1.4, cognitive weight: 1.0, social weight: [1.0, 1.5]

#### Conclusions

My feeling about the inertia is, that the closer you are to 0, the more chaotic turns particles make.

## Genetic Algorithms
Or memetic algorithm

Try the one from scipy first (also try some further algorithms form there)
https://docs.scipy.org/doc/scipy/reference/optimize.html

## Deep Reinforcement Learning

### Concept & Main Words

- An agent interacts with its enviornment through trial and error to find an optimum solution.
- The agent is rewarded for its decisions. The reward hypothesis states all goals can be described as a maximization of a cumulative reward.
- The policy $\pi$ is the decision making process of the agent. Given a state, the policy outputs an action or a probability distribution of actions.
- The goal is to find an optimal policy $\pi^*$ that solves the problem and leads to the best cumulative reward.

There are two types of RL methods:

1. Policy-based methods: directly learn the policy that leads to the best result (increase this weight when in this state)
2. Value-based methods: Give each state a value and learn the policy from there (this shape has value -5, which action to take to go to state with value -4)

### Value based methods

The agent learns a value function (Q or V) that maps a state to the expected value of
that state which is defined as the expected discounted return if starting from
the state. Or said differently: How much reward can it get starting from that state.

In value based methods, we need to define a policy by hand (usually something simple such as greedy policy) and the value function is the neural network that we learn.

#### Value Function Types

There are two types of value based methods depending on how we define the value function:

- **State value function (V)**:
  State value function outputs the expected return when starting from this sepcific state and follow the policy thereafter.

- **Action value function (Q)**:
  For each state & action pair the action value function calculates the expected return if the agent starts in one state and then takes that action.

**Bellman equation**
Instead of always calculating the value chain to the optimum goal to determine the state value or action value, one can assume that the value of a certain state is given by the sum of the immediate reward (going to the next state) + the value of the next state

$$ V(s) = E[R_{t+1}+\gamma * V(S_{t+1})|S_t=s] $$

Where $\gamma$ is the discount rate.

#### Learning strategies

- **Monte Carlo**:
  Uses an entire episode to learn.

Update value function ($V_{new}(s_t)$) after whole episode with $V_{old}(S_t) + learning_rate * (G_t - V_{old}(S_t))$ where $G_t$ is the return calculated from a whole episode.

- **Temporal Difference**:
  Uses single steps to learn and update the value function after each step.

Update value function ($V_{new}(s_t)$) after each step and use the current estiamte for the discounted value of the next state with $V_{old}(S_t) + learning_rate * (R_{t+1} + \gamma V_{old}(S_{t+1}) - V_{old}(S_t))$.

#### Q-Learning

Off-policy (use a different policy for acting and updating), value-based (training a value function), temportal difference method.

- Q-function: action-value function, Q stands for quality of a certain action for a given state
- value of a state: cumulative reward the agent gets when it starts with this state
- reward: feedback after performing an action
- Q-table: contains a row for each possible state and a column for each possible action (left, right, up, down). During learning the Q-table is adapted and an optimum policy is found.

**Algorithm**:

1. Initialize Q-table with random values or just zeros
2. Choose an action using a certain policy, e.g. $\epsilon$-greedy strategy - that is: with probability 1-$\epsilon$ the agent does exploitation (take route of highest state-action pair) and with the probability $\epsilon$ it does exploration (try random action). $\epsilon$ is reduced over time so that at the beginning a lot of exploration is done and then only exploitation.
3. Take action ($A_t$) and observe reward ($R_{t+1}$) and next state ($S_{t+1}$).
4. Update the Q-table using the temporal difference equation above with the discounted Q values of the next state.

In the end the optimal policy is the one that has the maximum Q-value for each state.

#### Deep Q-Learning

Now, creating a table that represents the whole state and action space is
practicle for small state spaces where we are dealing with < a few thousand
options. However, in many cases this is not possible (such as in the case of
more complex games). In this case we are approximating the Q-Table with a neural
network.

The neural network architecture takes the state (e.g. in form of a picture) as
input, adds some convolutional layers and outputs over some fully connected
layers the probability for a given action (left, right, shoot).

- Always reduce complexity (e.g. take out colors if they are not necessary)
- To add temporal information multiple stacks can be added (e.g. to track motion of a ball).
- Updating the weights of the NN now works as usual via a loss function that compares prediction and target and uses gradient descent to update the weights
- **Experience Replay**:
  Save experienced states to reuse them during training instead of calculating them and then discard them (learn from same experience multiple times). This can be important so that the NN doesn't forget older experiences
- **Fixed Q-target**:
  The problem is now that with one NN we cannot independently update one state without changing the target value (that was previously used in the Bellman equation). We therefore fix the parameters for the target value (or have a second NN) to prevent updating with an already updated NN.
- **Double DQN**:
  We don't have enough information on the best action of the next state anymore since we cannot simply use the state with the highest Q-value. The solution is to use two NNs to decouple the action selection from the target Q-value generation. (1. Use DQN to select best action that leads to the next state, 2. Use target network to calculate the Q-value of that next state.). Double DQNs reduce the risk of overestimating Q-values and allow faster and more stable training.

Learning for DQL first performs actions and stores the experiences. Only then the training happens from the action samples via gradient descent.

### Policy-based methods

The policy itself is a neural network that is directly trained.

The input of our netweork is now again a state and the NN outputs a probability
distribution to perform a certain action.

- In the learning process, we try to tweak the NN parameters in such a way that
  beneficial steps are taken more frequently.
- To determine if the policy is good or not, we define an objective function
  ($J(\Theta)$) that determines the performance of the agent given a trajectory
  (state action sequence without considering the reward) and outputs the expected
  cumulative reward. The goal of the policy-based method is to find a
  $\Theta$, a policy that maximizes the return in form of the objective
  function ($max_\Theta(J(\Theta))$)

$$J(\Theta) = \Sigma_i P(\tau; \Theta) R(\tau)$$

where $\tau$ is a trajectory, R the reward for a given trajectory and $\Theta$
the policy. Thus the objective function is defined as the sum of all rewards
weighted by the probability distribution over all trajectories.

#### Policy Gradient

Policy gradient uses a gradient ascent algorithm to optimize the objective function (optimize policy to maximize $J(\Theta)$). The policy is then updated via

$$\Theta + learningrate \nabla_\Theta J(\Theta)$$

However, we cannot calculate the true gradient of the objective function because it would involve calculating all the trajectories.

Policy Gradient Theorem: The policy gradient is defined by

$$\nabla_\Theta J(\Theta) = E_\pi[\nabla_\Theta log(\pi_\Theta (a_t|s_t))R(\Theta)]$$

The algorithm is then

1. Use the current best policy ($\pi_Theta$) to collect one or multiple trajectories ($\tau$)
2. Use the episode to estimate the gradient $\nabla J(\Theta)$
3. Update the weights of the policy with gradient ascent ($\Theta + learningrate \nabla J(\Theta)$)

#### Proximal Policy Gradient

#### Policy based optimization (PBO)

Viquerat et al.:  https://github.com/jviquerat/pbo

### Training Environments

Many implementations use OpenAI gym environements which are the defacto standard
for implementing environments in RL at the moment. Costum environements can be
implemented that inherit from the abstract gym.Env class. For more on this:

https://www.gymlibrary.dev/content/environment_creation/

For reinforcement learning algorithms it would be best to wrap shape generation
etc. into this class.

Continuous spaces can be realized with the Box() class instead of using Discrete. 

### Solver Libraries

Highly optimized package for proximal policy gradient optimization:

- https://www.samplefactory.dev
- https://stable-baselines3.readthedocs.io/

Best practice on using 
https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html


## Hyperparameter Tuning

Libraries:

- Optuna (optuna.org)
- Tensorboard (to observe parameters):

```
tensorboard --logdir=<path-to-logdir>
```
