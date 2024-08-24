# UPenn’s Perception-Action-Learning Lab: A Summary of Three Reinforcement Learning Papers

# How Far I’ll Go: Offline Goal-Conditioned RL via F-Advantage Regression
Yecheng Jason Ma, Jason Yan, Dinesh Jayaraman, Osbert Bastani
[paper](https://arxiv.org/pdf/2206.03023)

### Goal: 
Perfect offline goal-condition RL would mean the ability to learn any task from any relevant dataset with no real-world interaction (which can be much less efficient). Thus, we want to learn real-world manipulation purely from offline data by formulating goal-conditioned RL as a state-occupancy matching problem, with success displayed in a turning valve to various angles task

### Context:

#### [Goal-Conditioned RL](https://arxiv.org/pdf/2201.08299): 
In offline GCRL, we have a static dataset of transitions $(s,a,r,...)$ with goal $g$ (if the data is not generated from a goal-directed agent, then $g$ is randomly drawn from $p(g)$, our desired goals). 

We have extra tuple inputs to the problem:
- $G$, representing the space of goals for each task (Goals can be images, natural language, expected returns, etc)
- $p_g$, representing the desired goal distribution of env, 
- $\phi$: $S \rightarrow G$ (identity when current state is the goal), 
Also, R is defined with respect to G, and $\pi$ maximizes R over goal dist

Our overall loss function is the sum of losses for each goal multiplied by the weights for each goal. Luckily, each goal, given they’re in the same env, shares the same state/action dimensions, reward computing, and structure, so we can learn and share features across tasks. 

#### [State-occupancy matching]()
In state-occupancy matching, we try to make equal the amount a state is visited between distinct distributions (for example, as applied in GoFar below, between the state distributions of various tasks). The intuition here is to encourage the use of similar ‘skills/patterns’ between our ideal offline distribution and our learned policy action distribution to accelerate learning.

### Algorithm Summary: 
GoFar is a regression-based offline goal-conditioned RL algorithm that formulates learning as a state-occupancy matching problem between a dynamic-abiding ‘imitator’ agent and an ‘expert’ agent that teleports to the goal.

### Benefits/Advantages: 
1. GoFar doesn’t require Hindsight Goal Relabeling (HER), where we relabel goals to states achieved instead of original goals. GoFar alleviates sparse rewards in previous GCRL because ti tried to maximize state similarity to another distribution rather than rewards directly, and updating the policy as if data is coming from an optimal policy.

2. GoFar has uninterleaved optimization for value and actor, so we don’t need to train the policy until the Value function converges. 

3. GoFar reduces RL to a weighted regression problem, which gives stronger theoretical guarantees than regular RL.

4. The Dual-value objective does not depend on actions. This allows for a goal-only conditioned planner for zero-shot knowledge transfer to new domains of the same task and the same goal space. Moreover, we can use GoFar to learn goal-conditioned planners – we can plan low-level subgoals and enable low-level controllers for distant goals not previously designed.

### First objective

$Min_\pi D_{KL}(d^\pi(s;g)||p(s;g))$

Where:
$\pi$: policy
$D_{KL}$: Kullback-Leibler Divergence, measures the difference between two probability distributions (essentially relative entropies compared)
$D_{KL}(P||Q) = Σ P(x) * log(P(x)/Q(x))$

$d^\pi(s;g)$: goal-conditioned state-occupancy distribution of $\pi$
$p(s;g)$: distribution of states that satisfy g

Essentially, this quantifies the relative difference in entropies of the distribution of how much the policy is occupying goal states relative to the distribution of goal states. The actual ‘dynamics-abiding’ agent (left of ||) is trying to imitate the ‘teleporting’ agent (right of ||).

However, this depends on actions, but we want an objective independent of policy. Thus we use f-divergence regularization to get our dual problem:

### Dual problem Optimization
$\min_{V(s,g) \geq 0} (1-\gamma)\mathbb{E}{(s,g)\sim\mu_0,p(g)}[V(s;g)] + \mathbb{E}{(s,a,g)\sim d^O}[f^*(R(s;g) + \gamma\mathcal{T}V(s,a;g) - V(s;g))] $

This is the minimum over nonnegative Values of the discounted expectation over initial states and goals of the Value plus expectation over the (empirical) offline distribution of the convex conjugate of f-divergence of advantage

With definitions:
- Convex conjugate: for $f(x)$, $sup_x[<y,x> - f(x)]$ (convert to unconstrained by implicitly satisfying bellman flow constraint (how))
- F-divergence: the difference between two prob distributions (similar to in the context of state-occupancy matching (implemented as Chi-squared divergence ($\int (P(x) - Q(x))^2 / Q(x)$)
- Advantage: reward of this action plus discounted next state EV (from this action) - current state EV (Measures how good or bad this action is compared to the average action)

Overall, we find V that minimizes both EV of initial states/goals and f-divergence of the offline/trained distributions using the advantage function (which ensures learned policy is close to offline data actions). This allows for finding optimal value functions with only offline data. 

We then use a self-supervised regression update for the policy:

### Policy Updating
$\max_{\pi} \mathbb{E}{g\sim p(g)}\mathbb{E}{(s,a)\sim d^O(s,a;g)}[f'_(R(s;g) + \gamma\mathcal{T}V^(s,a;g) - V^*(s;g)) \log \pi(a|s,g)] $

This finds the maximum over policies of expectation over goals from goal distribution given expectation over states and goals from offline distribution of derivative of the convex conjugate of [(f-divergence function of advantage) + log probability of policy taking action given state and goal. It finds the policy to maximize the expected f-advantage weighted log prob of actions in the offline dataset. It uses weighted behavioral cloning, where high-advantage actions get more weight.

The dual-optimal advantage, $R(s; g) + γT V ∗ (s, a; g) − V ∗ (s; g)$ is also called f-advantage (hence f-advantage regression)

With this setup, no hindsight relabeling is needed, and the optimal value is learned without learned policy interleaving (why?). Essentially, this Reformulates RL as supervised learning!

### Algorithm
Run-through:
1. Train discriminator-based reward
2. Train dual value function V*(s;g)
3. Train policy $\pi$ via f-Advantage Regression

### Key Takeaways/Innovations
With GoFar, we can learn exclusively from offline behavior via f-divergence to optimize the advantage and similarity of actions between the offline optimal ‘teleporter’ policy and the learned policy. The dual divergence objective function allows for unconstrained optimization, and the formulation into supervised learning gives all the theoretical SL advantages. 


# VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pretraining
Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, {Vikash Kumar, Amy Zhang}
[paper](https://arxiv.org/pdf/2210.00030)

### Goal: 
Ideally, we could specify real-world task reward functions by specifying goal images and automating the rest, but simple pre-trained visual representations are not effective reward functions as embedding distance to goal image. VIP thus offers offline learning from any domain or type of data through a universal visual embedding-based reward evaluator learned entirely from out-of-domain data.

### Summary
VIP is a self-supervised pre-trained visual representation to generate dense/smooth embedding-based reward functions for unseen tasks. We do so using a novel implicit time contrastive objective. We can then use this learned reward function for goal-image-specified downstream tasks.

#### Key Innovation: 
Instead of solving impossible direct policy learning from out-of-domain action-free vids, we solve the dual problem of goal-conditioned value function learning, which can be trained without actions and entirely self-supervised.

VIP is suitable for pretraining on out-of-domain videos with no robot action labels!

### Context

#### [Time-contrastive learning](https://arxiv.org/abs/1704.06888): 
In time contrastive learning, we take in a video and try to produce embedding values for each frame representing the distance to a goal frame. Frames closer together are more likely to be similar in the embedding space. Moreover, we want the embedding to be smooth between frames, so there should be a similarity metric in embedding space between the current frame and the goal frame (so following trajectory the transitions in value should be smooth)

We treat segments close in time as positive examples and segments far in time as negative examples, then train a model to distinguish between them. We use a learned encoder to take in images and output K in the embedding dimension. (Clarify this).

We evaluate learned representations by constructing a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) from it defined by:
$M(\phi) := (\phi(O), A, R(o_t, o_{t+1}; \phi, g), T, \gamma, g)$
Where $\phi(O)$ are the embedded observations

The reward function is goal-embedding distance difference: 
$R(o_t, o_{t+1}; \phi, \{g\}) := S_\phi(o_{t+1}; g) - S_\phi(o_t; g) := (1 - \gamma)S_\phi(o_{t+1}; g) + (\gamma S_\phi(o_{t+1}; g) - S_\phi(o_t; g))$

Where $S_\phi(o_t; g) := - \|\phi(o_t) - \phi(g)\|_2$ which is distance in the representation space

Essentially the reward function measures progress toward the goal in the representation space, and we include a reward shaping term (discounted distance of next observation to goal). The policy $\pi$ maps embedded observations R^K into actions A.

### VIP Algorithm Setup

#### Self-supervised value learning from human videos
We start with the KL-regularized offline RL objective of learning from human videos:
    - $\max_{\pi^H, \phi} \mathbb{E}_{\pi^H}\left[\sum_t \gamma^t r(o; g)\right] - D_{KL}(d^{\pi^H}(o, a^H; g) \| d^D(o, \tilde{a}^H; g))$

This is the maximum over (policy of human actions $pi^H$ and phi) of expectation over $pi^H$ of (sum over t of t-discounted reward) - Kullback-Liebler Divergence of (distribution over observations and actions visited by $pi^H$ conditioned on G) given (distribution from dataset D with dummy action.
Essentially, this maximizes expected reward while keeping learned policy close to distribution.

We then use [Fenchel duality](https://en.wikipedia.org/wiki/Fenchel%27s_duality_theorem) to derive a dual optimization problem over value function:
    - $\max_\phi \min_V \mathbb{E}_{p(g)}\Big[(1 - \gamma)\mathbb{E}_{\mu_0(o;g)}[V(\phi(o); \phi(g))] + \log \mathbb{E}_{(o,o';g)\sim D} [\exp (r(o, g) + \gamma V(\phi(o'); \phi(g)) - V(\phi(o), \phi(g)))]\Big]$

Where:
- $\mu_0(o;g)$ is GC initial observations distribution 
- $D(o, o’:g)$ is GC distribution of 2 consecutive observations

This maximizes over phi and minimizes over V of Expectation over goal distribution of [(Expectation over initial observations distribution of V) + (log of expectation over GC distribution of 2 consecutive observation of exp(temporal difference error))]

It transforms the problem to maximize over the embedding function and minimize over V both the initial state value and the consecutive state errors. This thus eliminates the need for action labels from the first equation.

The dual objective doesn't need action labels, and uses simple reward function $r(o,g) = |(o == g) - 1$ (if we are at the goal state we get reward 0, and -1 otherwise)

#### Implicit Time-contrastive learning
Our objective is reinterpreted as a novel temporal contrastive learning where 
    - positive samples are initial frames and negative samples are middle frames between anchor and positive 
    - we have no repulsion of negatives, and we instead minimize temporal-difference error (V(o_t) - r + V(o_t+1))

### VIP Algorithm
We approximate $V*$ with negative L2 distribution in embedding space, and our final objective is: 
- $L(\phi) = E[\text{L2dist embedded observation and goal}]$

- Run-through:
1. With human videos and $\phi$, 
2. We sample sub-trajectories from the video dataset and treat initial/last frames as samples from goal and initial-state distributions
3. Compute L($\phi$)
4. Update $\phi$ with SGD


# DrEureka (Domain Randomization Eureka): Language Model Guided Sim-To-Real Transfer
Yecheng Jason Ma, William Liang, Hung-Ju Wang, Sam Wang, Yuke Zhu, Linxi “Jim” Fan, Osbert Bastani, Dinesh Jayaraman
[paper](https://arxiv.org/pdf/2406.01967)

### Goal:
Use LLMs to automate the design of simulation to real learning by automating the process of iteratively designing reward functions and adjusting sim params until convergence

### Summary
Given the physics simulation of a task, output reward functions and domain randomization distributions for automated sim2real transfer

We take in task and safety instructions and environment source code, and run [Eureka](https://eureka-research.github.io/) to generate a regularized reward function (what is this) and policy. We then test the policy under different sim conditions to build reward-aware physics priors and provide these to LLM to generate a set of domain randomization parameters. Lastly, we use synthesized reward and DR parameters to train policies for the real world.

### Context
[Domain Randomization](https://arxiv.org/pdf/2110.03239): applies randomization over distribution of simulation physical params so learned policy is robust against perturbance and better sim2real transfer. We both select the set of physics params ${p} \subset P$ and the randomization range for each of the chosen params. The step to choose the right parameter distribution for best sim-2-real transfer is typically manually tuned by humans (requires reasoning etc friction or restitution (amount of kinetic energy lost in collisions) for surfaces). 

[Eureka](https://eureka-research.github.io/): 
Run-through:
1. Given environment code as input
2. For a predefined number of iterations (3 - 5)
3. Generate reward functions, 
4. Evaluate the performance of multiple GPU-accelerated agents on different reward function candidates 
5. Adjust reward functions hyperparams and structures based on that info, etc. 
6. Return the final reward function

### Formalization

Markov Decision Process $M = (S, A, T)$ 
Where S is the environment state space, A is the action space, and T is the state transition function

Our true task objective $F: \Pi \rightarrow R$, policy to scalar performance eval

A sim2real algorithm for reward design + Domain Randomization takes M and a list of tasks $l_{task}$, and outputs reward fun R and distribution over transition functions $\Tau$.

Our policy learning algorithm A takes $(M, \Tau, R)$ and outputs policy $\pi $

The final policy is evaluated on unknown true MDP (real world) M*
$f* := F_M*(\pi)$

The goal of sim2real is to design physics params P and reward function R to maximize f*:
$max_{\Tau, R} F_M * (A(M, \Tau, R))$

### Algorithm

The Algorithm consists of three stages. 

#### Stage 1: LLM synthesizes reward functions

We use a modified Eureka which includes safety instructions in the prompt.
Run-through of modified Eureka
1. Given task description, safety instruction, RL algo A, env code M, LLM, fitness function F, and initial prompt; hyperparameters search iters N and batch size K
2. For N iterations (3 - 7)
3. Sample reward functions from LLM
4. Train policies in simulation using policy learning algo A(M, R) 
5. Evaluate policies in simulation using fitness function F
6. Reflect on the reward function with the best simulation performance
7. Update best reward function and policy (and equivalent fitness value)
8. At the end we output the final reward function and policy

#### Stage 2: Create RAPPs for DR

We use the policy + reward fun from Eureka to create a sampling range for physics params “reward aware physics priors (RAPP)”. We do so by evaluating policy on perturbed sim dynamics in simulation, which greatly narrows down possible DR configuration search range. For each DR param, we compute a lower and upper bound of feasible values for training.

Run-through:
1. Given policy, simulator, success criteria F, DR parameters P, and respective search vals R (what are search vals for DR)
2. For each DR param p in P (3 - 9)
3. Low and upper limits neg and pos infinity
4. For each search value r in R (5 - 8)
5. Set sim randomization param p to r
6. Evaluate policy in sim and record trajectory \tau
7. Evalue \tau on success criteria F
8. Set the parameter lower limit to the maximum of r and lower limit and upper limit to the minimum of r and upper limit
9. Output lower and upper limits for each param p

#### Stage 3: Generate Reward functions and evaluate policies

LLM uses resulting “reward aware physics priors (RAPP)” to generate DR config candidates $\Tau_{1..m}$
1. Given the DR configuration candidate, generated reward function, and environment simulator, 
2. Retrain policies suitable for real-world deployment using reinforcement learning on each DR config candidate
3. Evaluate all policies pi in the real world and choose the ‘best performing’