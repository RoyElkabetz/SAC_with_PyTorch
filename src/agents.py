import os
import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks import Actor, Critic, Value


class Agent:
    def __init__(self, gamma, alpha, beta, state_dims, action_dims, max_action, fc1_dim, fc2_dim,
                 memory_size, batch_size, tau, update_period, reward_scale, warmup, reparam_noise_lim,
                 name, ckpt_dir='tmp'):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.max_action = max_action
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_period = update_period
        self.tau = tau
        self.reward_scale = reward_scale
        self.warmup = warmup
        self.reparam_noise_lim = reparam_noise_lim
        self.name = name
        self.ckpt_dir = ckpt_dir

        model_name = f'{name}__' \
                     f'gamma_{gamma}__' \
                     f'alpha_{alpha}__' \
                     f'beta_{beta}__' \
                     f'fc1_{fc1_dim}__' \
                     f'fc2_{fc2_dim}__' \
                     f'bs_{batch_size}__' \
                     f'buffer_{memory_size}__' \
                     f'update_period_{update_period}__' \
                     f'tau_{tau}__'

        self.model_name = model_name
        self.learn_iter = 0
        self.full_path = os.path.join(self.ckpt_dir, self.model_name)

        # init replay buffer
        self.replay_buffer = ReplayBuffer(self.memory_size, self.state_dims, self.action_dims)

        # init actor network
        self.actor = Actor(self.alpha, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim, self.max_action,
                           self.reparam_noise_lim, name='Actor', ckpt_dir=self.ckpt_dir)

        # init critic networks
        self.critic_1 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                               name='Critic_1', ckpt_dir=self.ckpt_dir)
        self.critic_2 = Critic(self.beta, self.state_dims, self.action_dims, self.fc1_dim, self.fc2_dim,
                               name='Critic_2', ckpt_dir=self.ckpt_dir)

        # init value network
        self.value = Value(self.beta, self.state_dims, self.fc1_dim, self.fc2_dim,
                           name='Value', ckpt_dir=self.ckpt_dir)

        # init target value network
        self.target_value = Value(self.beta, self.state_dims, self.fc1_dim, self.fc2_dim,
                                  name='Target_Value', ckpt_dir=self.ckpt_dir)

        # hard network parameters update
        self.update_parameters(tau=1)

    def choose_action(self, state, deterministic=False, reparameterize=False):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        if deterministic:
            # deterministic action defined by the mean
            mu, _ = self.actor.forward(state)
            actions = mu
        else:
            # stochastic action is sampled from the normal distribution
            actions, _ = self.actor.sample_normal(state, reparameterize)
        return actions.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        return self.replay_buffer.store_transition(state, action, reward, state_, done)

    def load_batch(self):
        states, actions, rewards, states_, done = self.replay_buffer.load_batch(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.bool).to(self.actor.device)
        return states, actions, rewards, states_, done

    def update_parameters(self, tau=None):
        # update the target value network parameters
        if tau is None:
            tau = self.tau
        value_state_parameters = self.value.named_parameters()
        target_value_state_parameters = self.target_value.named_parameters()

        value_state_dict = dict(value_state_parameters)
        target_value_state_dict = dict(target_value_state_parameters)

        for item in value_state_dict:
            value_state_dict[item] = tau * value_state_dict[item].clone() + \
                                     (1 - tau) * target_value_state_dict[item].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_model(self):
        # saving all networks
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_model(self, gpu_to_cpu=False):
        # loading all networks
        self.actor.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.critic_1.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.critic_2.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.value.load_checkpoint(gpu_to_cpu=gpu_to_cpu)
        self.target_value.load_checkpoint(gpu_to_cpu=gpu_to_cpu)

    def learn(self):
        # learns only after warmup iteration and when there is at least a single batch in buffer to load
        if self.learn_iter < self.warmup or self.learn_iter < self.batch_size:
            return

        states, actions, rewards, states_, done = self.load_batch()

        # optimize the value network
        actions_, log_probs = self.actor.sample_normal(states, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_)
        q2_new_policy = self.critic_2.forward(states, actions_)
        q_min = T.min(q1_new_policy, q2_new_policy).to(self.value.device)
        q_min = q_min.view(-1)
        v = self.value.forward(states).view(-1)
        v_ = self.target_value.forward(states_).view(-1)
        v_[done] = 0.0

        target_value = q_min - log_probs
        value_loss = 0.5 * F.mse_loss(v, target_value)
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # optimize the actor network
        actions_, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_)
        q2_new_policy = self.critic_2.forward(states, actions_)
        q_min = T.min(q1_new_policy, q2_new_policy).to(self.value.device)
        q_min = q_min.view(-1)

        self.actor.optimizer.zero_grad()
        actor_loss = T.mean(log_probs - q_min)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # optimize the critic networks
        q_ = self.reward_scale * rewards.view(-1) + self.gamma * v_
        q1_old_policy = self.critic_1.forward(states, actions).view(-1)
        q2_old_policy = self.critic_2.forward(states, actions).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_)
        critic_loss = T.add(critic_1_loss, critic_2_loss)
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # soft update networks parameters
        if self.learn_iter % self.update_period == 0:
            self.update_parameters()
        self.learn_iter += 1

