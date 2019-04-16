import random
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from SharedAdam import SharedAdam


ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4
LEARNING_RATE = 0.001
GAMMA = 0.99
NUM_EPISODES = 1000
EPSILON_INIT = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1
INTERVAL_ASYNCUPDATE = 32
INTERVAL_TARGET = 100
PROCESS_NUM = 4


def predict(model, state):
    '''output the q values of different actions'''
    # a batch contains a single example
    return model.forward(torch.Tensor(state).unsqueeze(0)).squeeze(0)


def get_model():
    model = nn.Sequential(
          nn.Linear(OBSERVATIONS_DIM,16),
          nn.ReLU(),
          nn.Linear(16, 16),
          nn.ReLU(),
          nn.Linear(16, ACTIONS_DIM),
        )

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         torch.nn.init.zeros_(m.weight)
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)
    # model.apply(init_weights)

    model.share_memory()
    return model

def epsilon(episodes):
    return max((0-1)/(0.6*NUM_EPISODES)*episodes + 1, 0) # declind from 1 to 0 linearly

def train(rank, lock, action_model, target_model, optimizer, T):
    env = gym.make('CartPole-v0')
    t = 0

    for episode in range(NUM_EPISODES):
        state = env.reset()
        for step in count(1):
            values = predict(action_model, state)
            # epsilon greedy exploration
            if random.random() < epsilon(episode):
                action = random.choice(range(ACTIONS_DIM))
            else:
                action = values.argmax().item()

            next_state, reward, done, info = env.step(action)

            if done:
                y = reward
            else:
                with torch.no_grad():
                    y = reward + GAMMA * predict(target_model, next_state).max()
            # Accumulate gradients
            loss = (y - values[action]).pow(2)
            lock.acquire()
            loss.backward()
            lock.release()
            state = next_state
            T.value += 1
            t += 1
            if T.value % INTERVAL_TARGET == 0:
                target_model.load_state_dict(action_model.state_dict())
            if t % INTERVAL_ASYNCUPDATE == 0 or done:
                lock.acquire()
                optimizer.step() # update parameters
                optimizer.zero_grad()
                lock.release()
            if done:
                print('Process {}, Episode {}, epsilon:{:.3f}, steps:{}'.format(
                    rank,
                    episode+1,
                    epsilon(episode),
                    step
                ))
                break


if __name__ == "__main__":
    # Initialize action-value model with random weights
    action_model = get_model()

    # Initialize target model with same weights
    target_model = get_model()
    target_model.load_state_dict(action_model.state_dict())

    optimizer = SharedAdam(action_model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()
    optimizer.share_memory() # in case a gradient is used multiple times

    T = mp.Value('i', 0) # These shared objects will be process and thread-safe
    lock = mp.Lock()

    processes = []
    for rank in range(PROCESS_NUM):
        # On Unix a child process can make use of a shared resource created
        # in a parent process using a global resource
        # But it's good practice to always pass as arguments explicitly
        p = mp.Process(target=train, args=(rank, lock, action_model, target_model, optimizer, T))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

