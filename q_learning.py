import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.buffer = []

    def size(self):
        return len(self.buffer)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, batch_size):
        sample_size = min(batch_size, self.size())
        return random.sample(self.buffer, sample_size)

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 64)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        actions_value = self.out(x)
        return actions_value


class QAgent():
    def __init__(self, state_dim, action_dim, memory_capacity, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.epsilon = 0.8
        self.epsilon_decay_rate = 1 - 1e-3
        self.epsilon_min = 0.05
        self.device = torch.device('cuda')
        self.target_net = QNet(state_dim, action_dim).to(self.device)
        self.net = QNet(state_dim, action_dim).to(self.device)
        self.memory = ReplayMemory(memory_capacity)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def get_action(self, state):
        self.epsilon = (self.epsilon - self.epsilon_min) * self.epsilon_decay_rate + self.epsilon_min
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = state.reshape((1, -1))
        state_tensor = torch.from_numpy(state).to(self.device)
        action = self.net(state_tensor).detach().cpu().numpy()
        return np.argmax(action[0])

    def train_on_batch(self, batch_size):
        sample = self.memory.sample(batch_size)
        states = []
        n_states = []
        actions = []
        rewards = []
        dones = []
        for i in range(len(sample)):
            item = sample[i]
            state, n_state, action, reward, done =  item
            states.append(state.reshape((1, -1)))
            n_states.append(n_state.reshape((1, -1)))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        states = torch.FloatTensor(np.concatenate(states, axis=0)).to(self.device)
        n_states = torch.FloatTensor(np.concatenate(n_states, axis=0)).to(self.device)
        actions = torch.LongTensor(np.array(actions, dtype=np.int32).reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)
        q_eval = self.net(states).gather(1, actions)  # shape (batch, 1)
        q_next = self.target_net(n_states).detach()     # detach from graph, don't backpropagate
        # print(q_eval.size(), q_next.size())
        action_ = self.net(n_states).detach().max(1)[1].view(-1, 1)
        q_target = rewards + self.gamma * q_next.gather(1, action_) * (1 - done)   # shape
        # print(q_target.size())
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def switch_nets(self):
        action_dict = self.net.state_dict()
        self.net.load_state_dict(self.target_net.state_dict())
        self.target_net.load_state_dict(action_dict)

    def replace_weights(self, tau=0.01):
        '''
        update_eval_net_params
        '''
        net_state = self.net.state_dict()
        t_net_state = self.target_net.state_dict()
        for name in t_net_state.keys():
            params = (1 - tau) * t_net_state[name] + tau * net_state[name]
            t_net_state[name] = params
        self.target_net.load_state_dict(t_net_state)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.make("Blackjack-v0")
    agent = QAgent(4, 2, 1000000, 0.995)
    total_rewards = []
    for epoch in range(10000):
        observation = np.array(env.reset(), dtype=np.float32)
        total_reward = 0
        for step in range(1000):
            action = agent.get_action(observation)
            # action = env.action_space.sample()
            n_observation, reward, done, _ = env.step(action)
            n_observation = np.array(n_observation, dtype=np.float32)
            if done and step < 20:
                reward -= 10
            agent.memory.add([np.copy(observation), np.copy(n_observation), action, reward, done])
            observation = np.copy(n_observation)
            total_reward += reward

            for i in range(4):
                agent.train_on_batch(256)
            agent.replace_weights()
            if done:
                total_rewards.append(total_reward)
                print(epoch, total_reward, np.mean(total_rewards[-1000:]))
                break
