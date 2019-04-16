import random
from itertools import count
from collections import deque

import gym
import torch
import torch.nn as nn
from torch.optim import Adam


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, *transition):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append(transition)

    def sample(self, count):
        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)

class Agent:
    def __init__(self, states_dim=4, actions_dim=2, lr=0.001, gamma=0.99,
        replay_memory_size=1000, target_update_freq=100, minibatch_size=32):

        self.steps = 0
        self.episodes = 0
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.minibatch_size = minibatch_size
        self.actions_dim = actions_dim

        # Initialize replay memory D to capacity N
        self.replay = ReplayBuffer(replay_memory_size)
        # Initialize action-value model with random weights
        self.action_model = self._get_model(states_dim, actions_dim)
        self.optimizer = Adam(self.action_model.parameters(), lr=lr)
        self.criteria = nn.MSELoss()
        # Initialize target model with same weights
        self.target_model = self._get_model(states_dim, actions_dim)
        self.target_model.load_state_dict(self.action_model.state_dict())

    def _get_model(self, states_dim, actions_dim):
        model = nn.Sequential(
              nn.Linear(states_dim, 16),
              nn.ReLU(),
              nn.Linear(16, 16),
              nn.ReLU(),
              nn.Linear(16, actions_dim),
            )
        return model

    def _predict(self, model, states, squeeze=False):
        if squeeze:
            # a batch contains a single training sample
            return model.forward(torch.Tensor(states).unsqueeze(0)).squeeze(0)
        else:
            return model.forward(torch.Tensor(states))

    @property
    def epsilon(self):
        # epsilon = max(0.995**self.episodes, 0.1)
        epsilon = max((0-1)/200*self.episodes + 1, 0) # declind from 1 to 0 linearly
        return epsilon

    def act(self, state):
        # epsilon greedy exploration
        if random.random() >= self.epsilon:
            q_values = self._predict(self.action_model, state, squeeze=True)
            action = q_values.argmax().item()
        else:
            action = random.choice(range(self.actions_dim))
        return action

    def _train(self):
        # 1. sample a batch
        # 2. compute MSE loss
        # 3. update action network
        sample_transitions = self.replay.sample(self.minibatch_size)
        random.shuffle(sample_transitions)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*sample_transitions)
        action_batch = torch.tensor(action_batch).unsqueeze(1) # 1D -> 2D
        predicted_values = self._predict(self.action_model, state_batch).gather(dim=1, index=action_batch)
        mask = 1 - torch.Tensor(done_batch)
        next_action_values = self._predict(self.target_model, next_state_batch).detach().max(dim=1)[0]
        target_values = torch.Tensor(reward_batch) + mask * self.gamma * next_action_values
        target_values = target_values.unsqueeze(1) # 1D -> 2D
        self.optimizer.zero_grad()
        loss = self.criteria(predicted_values, target_values)
        loss.backward()
        self.optimizer.step()

    def transition(self, state, action, next_state, reward, done):
        self.steps += 1
        if done:
            self.episodes += 1
            # reward = -200/\
        self.replay.add(state, action, next_state, reward, int(done))
        # train
        if self.replay.size() >= self.minibatch_size:
            self._train()
        # update target network if needed
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.action_model.state_dict())


def main(agent):
    env = gym.make('CartPole-v0')
    for episode in range(1000):
        state = env.reset()
        for step in count(1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.transition(state, action, next_state, reward, done)
            state = next_state
            if done:
                print('Episode {}, epsilon:{:.3f}, steps:{}'.format(episode+1, agent.epsilon, step))
                break


if __name__ == "__main__":
    main(Agent())
