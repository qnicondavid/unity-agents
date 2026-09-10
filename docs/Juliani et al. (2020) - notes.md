# Unity: A General Platform for Intelligent Agents (Juliani et al., 2020)

**What's wrong with existing simulated environments:**
- unrealistic visuals
- inaccurate physics
- low task complexity
- restricted agent perspective
- limited capacity for interaction among artificial agents

**Their big claim:** modern game engines are uniquely suited to act as general platforms for developing rich learning environments

**Motivation chain:** benchmarks have been beaten, new environments drive new algorithms, but environments are costly to build

**Game engines are the proposed fix:** complex, realistic, intuitive for the user, multi-platform

- environment = the space where the AI agent acts
- simulator = the platform that computes the environment

**Axes:**
- sensory complexity (processing visual, auditory and text-based data)
- physical complexity (interaction with the environment)
- task logic complexity
- social complexity (interaction and communication among multiple agents, both within the same population and between different groups)

**Things that a simulator needs:** speed/parallelism, flexible control

**A survey of existing simulators:**
1. environment (single, fixed)
2. environment suite
3. domain-specific platform
4. general platform

**A Unity Project consists of a collection of Assets:**
- Scenes: a type of Asset that defines an environment (level)
  - GameObjects: the objects inside a Scene (physical or purely logical)
    - components: attached to each GameObject, they determine its behavior and function

**What the Unity Editor offers:**
- create custom Scenes
- record local expert demonstrations
- record large-scale demonstrations

**The Unity ML-Agents Toolkit:** an open-source project which enables researchers and developers to create simulated environments using the Unity Editor

**The toolkit includes:**
- a set of example environments
- state-of-the-art RL (reinforcement learning) algorithms
- imitation learning
- add-ons

**Entities of the ML-Agents SDK:**
- Sensors
- Agents
- an Academy

Each Agent component contains a policy labeled with a behavior name. Any number of agents can have a policy with the same behavior name. These agents will execute the same policy and share experience data.

It is possible to increase the speed of Unity ML-Agents simulations up to one hundred times real time (varies based on the computational resources and the complexity of the environment).

**Obstacle Tower:** environment for deep RL
- 100 randomly generated floors, each with an increasingly complex floor layout (puzzles, obstacles, enemies, locomotion challenges)
- goal: reach the end room of each floor and ascend to the top floor without entering a fail-state

**Results:** baseline agents solve 5/100 floors on average after 20 million time-steps of training
- humans who have interacted with the environment for 5 minutes solve 15 floors on average
- expert players: 50 floors
- contest-winning agent: 19 floors on average

**The first future direction:** environments that evolve alongside agents

**Human-in-the-loop training:**
- imitating expert trajectories
- humans providing evaluative feedback to an agent
- humans manipulating the agent's observed states and actions

**Under-explored research problem:** training agents to be challenging to humans but not so dominant that the human does not engage in future contests

Unity makes it possible to study agent-human interaction at scale as humans play with or against an agent.

Training agents against many humans with different play styles will also improve the generalization and robustness of the learned policy.

**Future directions:**
- the toolkit is also meant for game developers who are not necessarily machine learning experts, but tuning hyperparameters may be insurmountable for a non-expert in some cases
- plan: intuitive UI tools for tuning algorithms (e.g., tweaking reward functions, defining observations and actions)
- improving the Unity engine and the ML-Agents Toolkit in both performance and breadth