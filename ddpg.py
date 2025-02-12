import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from random_process import OrnsteinUhlenbeckProcess

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=500):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[idx],
            self.action[idx],
            self.next_state[idx],
            self.reward[idx],
            self.not_done[idx]
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.l3 = nn.Linear(300, action_dim)
        self.bn3 = nn.BatchNorm1d(action_dim)

        self.max_action = max_action

    def forward(self, state):
        if state.size(0) > 1:
            a = torch.relu(self.bn1(self.l1(state)))
            a = torch.relu(self.bn2(self.l2(a)))
            a = torch.tanh(self.bn3(self.l3(a))) * self.max_action
        else:
            a = torch.relu(self.l1(state))
            a = torch.relu(self.l2(a))
            a = torch.tanh(self.l3(a)) * self.max_action

        return a
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        if state.size(0) > 1:
            sa = torch.cat([state, action], 1)
            q = torch.relu(self.bn1(self.l1(sa)))
            q = torch.relu(self.bn2(self.l2(q)))
            q = self.l3(q)
        else:
            sa = torch.cat([state, action], 1)
            q = torch.relu(self.l1(sa))
            q = torch.relu(self.l2(q))
            q = self.l3(q)
            
        return q


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, capacity, device="cpu"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=capacity)
        self.max_action = max_action
        self.device = device
        self.random_process = OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, mu=0, sigma=0.2)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # action = self.actor(state).cpu().data.numpy().flatten() + self.random_process.sample()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action.clip(-self.max_action, self.max_action)

    def train(self, batch_size=100, discount=0.99, tau=0.005, lr=1e-5):
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        next_action = self.actor_target(next_state)
        target_Q = reward + (not_done * discount * self.critic_target(next_state, next_action)).detach()
        current_Q = self.critic(state, action)

        critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        critic_optimizer.zero_grad()

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        actor_optimizer.zero_grad()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_loss.backward()
        actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "actor.pth")
        torch.save(self.critic.state_dict(), filename + "critic.pth")
        
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "actor.pth"))

    def reset(self):
        self.random_process.reset_states()
        