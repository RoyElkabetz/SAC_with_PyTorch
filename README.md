# SAC with PyTorch

This repo contains a PyTorch implementation of the Deep Reinforcement Learning algorithm Soft Actor Critic (SAC), as described in the original paper:

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (2018)



## Requirements and ROM installation

|Library         | Version |
|----------------|---------|
|`Python`        |  `3.8`  |
|`torch`         |  `1.8.1`|
|`gym`           | `0.18.3`|
|`numpy`         | `1.19.5`|
|`pybullet`      | `3.21`  |


## Folders and Files Description

### Folders

|Folder name       |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`src`             | All `.py` sourch files                                             |
|`tmp `            | A temporary file for results savings                               |
|`assets`          | Non-code relevant files like videos, pictures, etc                 |


### Files

|File name         |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`main.py`         | General main application for training/playing a SAC based agent    |
|`agents.py`       | Containing the SAc agent class                                     |
|`networks.py`     | Networks in used by agents (Actor, Critic and Value networks)      |
|`utils.py`        | General utility functions                                          |
|`buffer.py`       | A replay buffer class, used for offline training                   |



## Command Line API

You should run the `main.py` file with the following arguments:

|Argument             | Description                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------|
|`-train`             | Determine the agents mode, True=training or False=playing, default=False                      |
|`-gamma`             | Discount factor for the update rule, default=0.99                                             |
|`-epsilon`           | Initial epsilon value for the epsilon-greedy policy, default=1.0                              |
|`-lr`                | The DQN training learning rate, default=0.0001                                                |
|`-mem_size`          | The maximal memory size used for storing transitions (replay buffer), default=20000 (~ 6 GB RAM) |
|`-bs`                | Batch size for sampling from the replay buffer, default=32                                    |
|`-eps_min`           | Lower limit for epsilon, default=0.1                                                          |
|`-eps_dec`           | Value for epsilon linear decrement, default=1e-5                                              |
|`-replace`           | Number of learning steps for target network replacement, default=1000                         |
|`-algo`              | choose from the next algorithms: `DQNAgent`, `DDQNAgent`, `DuelingDQNAgent`, `DuelingDDQNAgent`, default=`DQNAgent`|
|`-env_name`          | choose from the next Atari environments: `PongNoFrameskip-v4`, `BreakoutNoFrameskip-v4`, `SpaceInvadersNoFrameskip-v4`, `EnduroNoFrameskip-v4`, `AtlantisNoFrameskip-v4`, default=`PongNoFrameskip-v4`        |
|`-path`              | Path for loading and saving models, default='models/'                                         |
|`-n_games`           | Number of games for the Agent to play, default=1                                              |
|`-skip`              | Number of environment frames to stack, default=4                                              |
|`-gpu`               | CPU: '0', GPU: '1', default='0'                                                               |
|`-load_checkpoint`   | Load a model checkpoint, default=False                                                        |
|`-render`            | Render the game to screen ? True/False, default=False                                         |
|`-monitor`           | If True, a video is being saved for each episode, default=False                               |


## Training and Playing
- Training a DuelingDDQN agent from scratch for 400 games

```text
python main.py -n_games 400 -algo 'DuelingDDQNAgent' -train True
``` 

- Training a DDQN agent from checkpoint (if exist) for 30 games with epsilon=0.2 and batch size of 64

```text
python main.py -n_games 30 -algo 'DDQNAgent' -load_checkpoint True -epsilon 0.2 -bs 64 -train True
```

- Playing 10 games with a saved DQN agent checkpoint using a deterministic policy (epsilon=0), render to screen and save as a video

```text
python main.py -n_games 10 -algo 'DQNAgent' -load_checkpoint True -epsilon 0.0 -eps_min 0.0 -render True -monitor True
```

- Playing 5 games with an untrained DuelingDQN agent using an epsilon-greedy policy with epsilon=0.2 and render to screen

```text
python main.py -n_games 5 -algo 'DuelingDQNAgent' -epsilon 0.2 -eps_dec 0.0 -render True -monitor True
```

**Notes:**
- If training from checkpoint, the agent also upload previous saved scores, steps and epsilon arrays, such that the training process continues from where it stopped.
- For playing with an agent using an epsilon-greedy policy with some specific epsilon (i.e 0.1), you need to set eps_dec=0.0 (-eps_dec 0.0). Otherwise, epsilon would get smaller at each step by the eps_dec value. 


## Reference

[1]  [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (2015)

[2]  [Modern Reinforcement Learning: Deep Q Learning in PyTorch Course - Phil Tabor](https://www.udemy.com/course/deep-q-learning-from-paper-to-code/) (great comprehensive course about DQN algorithms)


 

 
