# SAC_with_PyTorch

This repo contains a PyTorch implementation of the Deep Reinforcement Learning algorithm Soft Actor Critic (SAC), as described in the original paper:

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (2018)


## Background
In the first paper above (Human-level control through deep reinforcement learning (2005)) the authors state *"We set out to create a single algorithm that would be able to develop a wide range of competencies on a varied range of challenging tasks — a central goal of general artificial intelligence"*. Indeed, the main advantages in estimating the **Q-value function** using a Deep Neural Network (DNN) are, (1) An identical network can be used in a variety of very different games and sequential tasks, (2) The complexity of the training does not scale trivially with the size of the (state, action) space, which means that a very large (state, action) space can be modeled without a problem using a pretty small DNN (comparing to real-life applications solved using DNNs). In this repository, I followed the development of the DQN to DDQN and then to Dueling-DQN and Dueling-DDQN algorithms, and implemented all four of them as described in the papers. My goal was less to make a clean and clear API for DQN algorithms rather than to gain some fundamental understanding of the basic concepts that drove the DRL field forward in the last few years.

## Requirements and ROM installation

|Library         | Version |
|----------------|---------|
|`Python`        |  `3.8`  |
|`torch`         |  `1.8.1`|
|`gym`           | `0.18.3`|
|`numpy`         | `1.19.5`|

### ROMs installation
After installing the gym library, in order to render the games from the Atari library you need to install the Atari ROMs following the next few steps:
1. Download and save the Atari ROM files from the next [url](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).
2. Extract from the downloaded `Roms.rar` file the two zip files `HC ROMS.zip` and `ROMS.zip`.
3. Open a Terminal window.
4. Run the next command in the terminal 
```text
python -m atari_py.import_roms path_to_folder\ROMS
```

Example: 
```text
python -m atari_py.import_roms C:\Users\ME\Downloads\Roms\ROMS
```

Note: if your default python version is different from the one you will be using in working with gym (i.e python 2 as default but you will be using python 3 ,use `python3` instead of `python` in step (4)).

## Folders and Files Description

### Folders

|Folder name       |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`models`          | saved checkpoints of DQN networks                                  |
|`papers `         | pdf files of the three papers my code is based on                  |
|`plots`           | plots of learning curves                                           |
|`scores`          | saved .npy scores, epsilon and steps files                         |
|`videos`          | saved videos of the agents playing                                 |

### Files

|File name         |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`main.py`         | general main application for training/playing a DQN based agent    |
|`agents.py`       | containing classes of DQN, DDQN, DuelingDQN and DuelingDDQN agents |
|`networks.py`     | networks in used by agents                                         |
|`utils.py`        | utility functions                                                  |
|`replay_memory.py`| replay buffer class, used for training the DQN agent               |
|`dqn_main.ipynb`  | general main application in a notebook format for training/playing |



## API

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

[2]  [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015)

[3]  [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (2016)

[4]  [Modern Reinforcement Learning: Deep Q Learning in PyTorch Course - Phil Tabor](https://www.udemy.com/course/deep-q-learning-from-paper-to-code/) (great comprehensive course about DQN algorithms)


 

 
