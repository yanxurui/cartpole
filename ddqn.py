import random
import torch
import dqn
from dqn import main


class Agent(dqn.Agent):
    def _train(self):
        # 1. sample a batch
        # 2. compute MSE loss
        # 3. update action network
        sample_transitions = self.replay.sample(self.minibatch_size)
        random.shuffle(sample_transitions)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*sample_transitions)
        action_batch = torch.tensor(action_batch).unsqueeze(1) # 1D -> 2D
        predicted_values = self._predict(self.action_model, state_batch).gather(dim=1, index=action_batch) # 2D
        mask = 1 - torch.Tensor(done_batch)
        next_action_batch = self._predict(self.action_model, next_state_batch).detach().argmax(dim=1, keepdim=True)
        next_action_values = self._predict(self.target_model, next_state_batch).detach().gather(dim=1, index=next_action_batch) # 2D
        target_values = torch.Tensor(reward_batch) + mask * self.gamma * next_action_values.squeeze(1)
        target_values = target_values.unsqueeze(1) # 1D -> 2D
        self.optimizer.zero_grad()
        loss = self.criteria(predicted_values, target_values)
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    main(Agent())
