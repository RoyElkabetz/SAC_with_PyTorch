import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_dims):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.is_full = False

        # Init buffer
        self.states = np.zeros((self.buffer_size, *state_dims), dtype=np.float32)
        self.states_ = np.zeros((self.buffer_size, *state_dims), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, *action_dims), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.done = np.zeros((self.buffer_size,), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.states_[self.ptr] = state_
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size

        if self.ptr == 0 and not self.is_full:
            self.is_full = True
            print('... Replay Buffer is full ...')

    def load_batch(self, batch_size):
        samples = np.random.choice(np.arange(self.buffer_size), batch_size, replace=False)
        states = self.states[samples]
        actions = self.actions[samples]
        rewards = self.rewards[samples]
        states_ = self.states_[samples]
        done = self.done[samples]

        return states, actions, rewards, states_, done
