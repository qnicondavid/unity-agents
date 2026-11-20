# Temperature - research notes

## Index

- [Intro](#intro)
- [Arrhenius](#arrhenius)
  - [Equation](#equation)
  - [Plot](#plot)
  - [Waiting time instead of speed](#waiting-time-instead-of-speed)
- [The weights as a reaction](#the-weights-as-a-reaction)
- [Learning rate and heat](#learning-rate-and-heat)
- [The training Arrhenius plot](#the-training-arrhenius-plot)
- [Adam](#adam)
- [PPO in ML-Agents](#ppo-in-ml-agents)
- [Scope](#scope)

## Intro

We analyze the energy as reaction progress increases.

```text
energy      transition state
  │            ╱‾‾╲
  │           ╱    ╲
  │ reactants╱      ╲
  │                  ╲____ products
  └──────────────────────── reaction progress
```

Along the way, there are some spikes called transition states.

At a given temperature, molecules don’t all have the same energy.
Temperature is the average kinetic energy of the molecules, but their energies are spread out.

The share of molecules above a barrier is about $\exp(-E_a/RT)$, so it grows fast as $T$ rises.

- $E_a$: the activation energy, the height of the barrier between reactants and products, in J/mol
- $\exp(x)$: $e^x$, with $e \approx 2.718$; for a negative $x$ it lies between 0 and 1, so it works as a share
- $R$: the gas constant, 8.314 J/(mol·K)

**Cold:** few molecules cross barriers, so things are slow.

**Hot:** many cross, so things are fast.

Temperature changes how much random jiggling there is.

## Arrhenius

### Equation

$$k = A \cdot \exp(-E_a / RT)$$

- $k$ is the rate constant: how fast the reaction goes. Bigger $k$ means faster.
- $A$ is the pre-exponential (frequency) factor: how often molecules collide the right way round, so how many attempts there are.

The equation reads: speed = attempts × share of attempts that clear the barrier.

### Plot

$$k = A \cdot \exp(-E_a / RT)$$

Take the logarithm:

$$\ln k = \ln A - (E_a/R) \cdot (1/T)$$

That’s a straight line, $y = c + m \cdot x$, with $y = \ln k$, $x = 1/T$ and slope $m = -E_a/R$.

So, $E_a = -m \cdot R$.

That’s how chemists measure a barrier they never see:

- measure $k$ at a few temperatures
- draw the line
- read $E_a$ from the slope

### Waiting time instead of speed

Our runs give a time (steps to 95), not a speed.

For a first-order reaction, the mean time to react is $\tau = 1/k$, so:

$$\ln \tau = -\ln A + (E_a/R) \cdot (1/T)$$

## The weights as a reaction

```text
loss
  │ ╲          ╱╲
  │  ╲  start ╱  ╲            ╱
  │   ╲______╱    ╲          ╱
  │                ╲________╱   ← better weights
  └───────────────────────────── weights
```

$$\text{new weights} = \text{old weights} - \text{learning rate} \times \text{gradient}$$

With the exact gradient there’s no randomness, like at temperature zero.

- Each step uses a minibatch of $B$ samples, not all the data.
- Minibatch gradient = true gradient + random error. That error is the “heat”.
- Bigger batch = less noise (variance $\propto 1/B$) = colder.

## Learning rate and heat

**(0)**

- $\hat{g}$: minibatch gradient
- $g$: true gradient
- $\xi$: error
- $\sigma^2$: variance of a single sample's gradient
- $\mathrm{lr}$: learning rate
- $B$: batch size
- $n$: training steps

**(1)**

$$\hat{g} = g + \xi$$

**(2)**

$$
\begin{aligned}
\text{new\_weights} &= \text{old\_weights} - \mathrm{lr} \cdot \hat{g} \\
&= \text{old\_weights} - \mathrm{lr} \cdot g - \mathrm{lr} \cdot \xi
\end{aligned}
$$

**(3)**

$$\mathrm{Var}(\xi) = \sigma^2 / B$$

Variance measures how spread out a random number is.

**(4)** Let $\mathrm{lr} \cdot \xi$ be called the kick.

$$\mathrm{Var}(\text{kick}) = \mathrm{Var}(\mathrm{lr} \cdot \xi) = \mathrm{lr}^2 \cdot \mathrm{Var}(\xi) = \mathrm{lr}^2 \cdot \sigma^2 / B$$

**(5)** Define the training time $t$ as $t = n \cdot \mathrm{lr}$.

**(6)**

$$
\begin{aligned}
\mathrm{Var}(\text{all kicks}) &= n \cdot \mathrm{Var}(\text{kick}) \\
&= n \cdot \mathrm{lr}^2 \cdot \sigma^2 / B \\
&= (t / \mathrm{lr}) \cdot \mathrm{lr}^2 \cdot \sigma^2 / B \\
&= t \cdot \mathrm{lr} \cdot \sigma^2 / B
\end{aligned}
$$

**(7)** In the Intro, it is mentioned that temperature changes how much random jiggling there is.

Now compare two training runs after the same progress $t$.
After $t$, the only difference between runs is how much random jiggling they collected.

- **Downhill part:** both runs have moved downhill by the same amount, $g \cdot t$.
- **Random part:** each run has collected its own random spread with variance $t \cdot \mathrm{lr} \cdot \sigma^2 / B$.

$$T \propto \mathrm{Var}(\text{all kicks}) / t = \mathrm{lr} \cdot \sigma^2 / B \propto \mathrm{lr} / B$$

Divide by $t$ because temperature is a rate, how much jiggling per unit of time, not the total.

## The training Arrhenius plot

What we have already derived:

$$\tau \propto \exp(E_a / RT)$$

For training, the energy becomes the loss:

$$\tau \propto \exp(\Delta L / T) \propto \exp(c \cdot \Delta L \cdot B / \mathrm{lr})$$

$c$ is a constant that depends on $\sigma^2$.

Take the log:

$$\ln(\tau) = \ln(\text{steps to escape}) = \text{const} + c \cdot \Delta L \cdot (B / \mathrm{lr})$$

That’s a straight line against $B/\mathrm{lr}$, with a slope proportional to the barrier $\Delta L$.
It’s the waiting-time line from the chemistry part, with $1/T$ replaced by $B/\mathrm{lr}$.

## Adam

ML-Agents' PPO uses Adam, which updates the weights in a different way: it divides each step by a running average of the gradient's size, so every step is about $\mathrm{lr}$ long.
Repeating the derivation for Adam leads to $T \propto \mathrm{lr} / \sqrt{B}$ instead of $T \propto \mathrm{lr} / B$.

## PPO in ML-Agents

**The x-axis.** Our target counts environment steps: the experience the agents collected, not the number of weight updates.
The scaling rules say that runs with the same temperature, $\mathrm{lr}/B$ with SGD or $\mathrm{lr}/\sqrt{B}$ with Adam, behave about the same when compared after the same amount of data. So steps to 95 is the right axis for comparing temperatures.

**The learning rate cools down.** ML-Agents lowers the learning rate linearly, to almost 0 at the end of the budget:

$$\mathrm{lr}(s) = \mathrm{lr}_{\text{start}} \cdot (1 - s / 500{,}000)$$

Every run has the same budget, so at any step all runs are cooled by the same factor. The starting temperature still ranks the runs.

## Scope

This is early work: we try to check whether the runs we already have are consistent with the temperature idea.
One goal addresses the question of hyperparameter importance, which is related to how hyperparameters interact, and here we may be able to combine $\mathrm{lr}$ and $B$ into a single parameter called temperature.
The implications for the other research questions are future work.

Script ideas so far:

1. **Temperature table.** For each ball and each pair of learning rate and batch size: the median steps to 95, the share of runs that never reach 95, and the gap between seeds 2 and 3.
