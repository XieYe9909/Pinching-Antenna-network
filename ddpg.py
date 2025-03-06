import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from random_process import OrnsteinUhlenbeckProcess

class ReplayBuffer:
    def __init__(self, channel_dim, location_dim, feature_dim, action_dim, max_size=500):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.channel = np.zeros((max_size, channel_dim))
        self.location = np.zeros((max_size, location_dim))
        self.feature = np.zeros((max_size, feature_dim))
        self.next_channel = np.zeros((max_size, channel_dim))
        self.next_location = np.zeros((max_size, location_dim))
        self.next_feature = np.zeros((max_size, feature_dim))

        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, channel, location, feature, next_channel, next_location, next_feature, action, reward, done):
        self.channel[self.ptr] = channel
        self.location[self.ptr] = location
        self.feature[self.ptr] = feature
        self.next_channel[self.ptr] = next_channel
        self.next_location[self.ptr] = next_location
        self.next_feature[self.ptr] = next_feature
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        # self.state[self.ptr] = state
        # self.action[self.ptr] = action
        # self.next_state[self.ptr] = next_state
        # self.reward[self.ptr] = reward
        # self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            self.channel[idx],
            self.location[idx],
            self.feature[idx],
            self.next_channel[idx],
            self.next_location[idx],
            self.next_feature[idx],
            self.action[idx],
            self.reward[idx],
            self.done[idx]
        )

class Abstractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Abstractor, self).__init__()

        self.l1 = nn.Linear(input_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)

        self.l2 = nn.Linear(300, 400)
        self.bn2 = nn.BatchNorm1d(400)

        self.l3 = nn.Linear(400, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, channel):
        if channel.size(0) > 1:
            a = torch.relu(self.bn1(self.l1(channel)))
            a = torch.relu(self.bn2(self.l2(a)))
            a = torch.sigmoid(self.bn3(self.l3(a)))
        else:
            a = torch.relu(self.l1(channel))
            a = torch.relu(self.l2(a))
            a = torch.sigmoid(self.l3(a))

        return a

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.l2 = nn.Linear(400, 600)
        self.bn2 = nn.BatchNorm1d(600)

        self.l3 = nn.Linear(600, 300)
        self.bn3 = nn.BatchNorm1d(300)

        self.l4 = nn.Linear(300, action_dim)
        self.bn4 = nn.BatchNorm1d(action_dim)

        self.max_action = max_action

    def forward(self, state):
        if state.size(0) > 1:
            a = torch.relu(self.bn1(self.l1(state)))
            a = torch.relu(self.bn2(self.l2(a)))
            a = torch.relu(self.bn3(self.l3(a)))
            a = torch.tanh(self.bn4(self.l4(a))) * self.max_action
        else:
            a = torch.relu(self.l1(state))
            a = torch.relu(self.l2(a))
            a = torch.relu(self.l3(a))
            a = torch.tanh(self.l4(a)) * self.max_action

        return a
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.l2 = nn.Linear(400, 600)
        self.bn2 = nn.BatchNorm1d(600)

        self.l3 = nn.Linear(600, 300)
        self.bn3 = nn.BatchNorm1d(300)

        self.l4 = nn.Linear(300, 1)

    def forward(self, state, action):
        if state.size(0) > 1:
            sa = torch.cat([state, action], 1)
            q = torch.relu(self.bn1(self.l1(sa)))
            q = torch.relu(self.bn2(self.l2(q)))
            q = torch.relu(self.bn3(self.l3(q)))
            q = self.l4(q)
        else:
            sa = torch.cat([state, action], 1)
            q = torch.relu(self.l1(sa))
            q = torch.relu(self.l2(q))
            q = torch.relu(self.l3(q))
            q = self.l4(q)
            
        return q


class DDPG:
    def __init__(self, channel_dim, feature_dim, location_dim, action_dim, max_action, capacity, device="cpu"):
        self.abstractor = Abstractor(channel_dim, feature_dim).to(device)

        state_dim = feature_dim + location_dim
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.channel_dim = channel_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(channel_dim, location_dim, feature_dim, action_dim, max_size=capacity)
        self.max_action = max_action
        self.device = device
        self.random_process = OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, mu=0, sigma=0.2)

    def select_action(self, channel, location):
        channel = torch.FloatTensor(channel).to(self.device)
        location = torch.FloatTensor(location).to(self.device)
        feature = self.abstractor(channel)
        state = torch.cat([feature, location], 1)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, batch_size=100, gamma=0.99, tau=0.005, lr=1e-5):
        channel, location, feature, next_channel, next_location, next_feature, action, reward, done = self.replay_buffer.sample(batch_size)

        channel = torch.FloatTensor(channel).to(self.device)
        location = torch.FloatTensor(location).to(self.device)
        feature = torch.FloatTensor(feature).to(self.device)
        next_channel = torch.FloatTensor(next_channel).to(self.device)
        next_location = torch.FloatTensor(next_location).to(self.device)
        next_feature = torch.FloatTensor(next_feature).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        current_Q = self.critic(torch.cat([feature, location], 1), action)
        next_action = self.actor_target(torch.cat([next_feature, next_location], 1))
        next_Q = self.critic_target(torch.cat([next_feature, next_location], 1), next_action).detach()
        target_Q = reward + (1 - done) * gamma * next_Q

        critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        critic_optimizer.zero_grad()

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        actor_optimizer.zero_grad()

        state = torch.cat([self.abstractor(channel), location], 1)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()

        abstractor_optimizer = Adam(self.abstractor.parameters(), lr=lr)
        abstractor_optimizer.zero_grad()

        abstractor_loss = -self.critic(state, self.actor(state)).mean()
        abstractor_loss.backward()
        abstractor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
    
    def save(self, filename):
        torch.save(self.abstractor.state_dict(), filename + "abstractor.pth")
        torch.save(self.actor.state_dict(), filename + "actor.pth")
        torch.save(self.critic.state_dict(), filename + "critic.pth")
        
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "actor.pth"))

    def reset(self):
        self.random_process.reset_states()
        