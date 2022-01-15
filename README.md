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
|`assets`          | Non-code related files like videos, pictures, etc                 |


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





    
   
    
    

   
    
    
    
    
    
    
    
    parser.add_argument('-monitor', type=bool, default=False,
                        help='If True, a video is being saved for each episode')




|Argument             | Description                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------|
|`-play`              | Choosing the mode of the agent, False for learning or True for playing and render to screen   |
|`-gamma`             | Discount factor for the update rule, default=0.99                                             |
|`-alpha`             | The Actor network learning rate                                                               |
|`-beta`              | The Critic and Value networks learning rate                                                   |
|`-fc1_dim`           | The dimension of the first Linear layer across all networks                                   |
|`-fc2_dim`           | The dimension of the second Linear layer across all networks                                  |
|`-memory_size`       | The Replay Buffer memory size                                                                 |
|`-batch_size`        | The batch size                                                                                |
|`-tau`               | The parameters update constant -- 1 for hard update, 0 for no update                          |
|`-update_period`     | The period for updating the networks weights                                                  |
|`-reward_scale`      | The scale factor for the rewards as written in the paper (exploration/exploitation parameter) |
|`-warmup`            | Number of transitions passes before learning starts                                           |
|`-reparam_noise_lim` | Lower limit of the reparametrization noise (the upper limit is hardcoded to 1.)               |
|`-n_games`           | Number of games / episodes                                                                    |
|`-env_name`          | The environment name, PyBullet or Gym                                                         |
|`-load_checkpoint`   | Load a model checkpoint, default=False                                                        |
|`-gpu_to_cpu`        | Load to a CPU a model that was trained on a GPU, set to True, else False                      |
|`-dir`               | Path for loading and saving models and plots                                                  |
|`-monitor`           | If True, a video is being saved for each episode (only if the ffmpeg package is installs)     |


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


 

 
