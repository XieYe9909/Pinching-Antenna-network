import numpy as np
import torch
import gym
import argparse
import matplotlib.pyplot as plt

from gym.wrappers import FlattenObservation
from ddpg import DDPG

env_name = "PinchingAntenna-v1"  # gym environment
mode = 'train'  # mode = 'train' or 'test' or 'plot'
gamma = 0.7  # discounted factor
tau = 0.005  # target smoothing coefficient
learning_rate = 1e-5
capacity = 1000  # replay buffer size
max_episode = 10000
num_iteration = 300
warmup = 50  # time without training but only filling the replay memory
batch_size =5  # mini batch size
max_action = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'data/'
log_interval = 100

num_waveguides = 2
num_users = 2

env = gym.make(id=env_name, num_waveguides=num_waveguides, num_users=num_users)
if isinstance(env.observation_space, gym.spaces.Dict):
    env = FlattenObservation(env)

channel_dim = 2 * num_waveguides * num_users
feature_dim = location_dim = num_waveguides
action_dim = env.action_space.shape[0]

agent = DDPG(channel_dim, feature_dim, location_dim, action_dim, max_action, capacity, device=device)
if mode == 'test':
    agent.load(path)
    for episode in range(max_episode):
        state = env.reset()[0]
        episode_reward = env.sum_rate

        for step in range(num_iteration):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            if done:
                print("Episode: {}, Total Reward: {}".format(episode, episode_reward))
                break

            state = next_state

elif mode == 'train':
    print("====================================")
    print("Collection Experience...")
    print("====================================")
    
    for episode in range(max_episode):
        loc_antennas_init = np.zeros(shape=(num_waveguides,))
        state = env.reset(loc_antennas=loc_antennas_init)[0]
        episode_reward = env.sum_rate
        agent.reset()

        for step in range(num_iteration):
            channel = state[:2 * num_waveguides * num_users].reshape(1, -1)
            location = state[2 * num_waveguides * num_users:].reshape(1, -1)
            feature = agent.abstractor(torch.FloatTensor(channel).to(device))
            feature = feature.cpu().data.numpy().flatten()

            if episode == 0 and step <= warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(channel, location)

            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            next_channel = next_state[:2 * num_waveguides * num_users].reshape(1, -1)
            next_location = next_state[2 * num_waveguides * num_users:].reshape(1, -1)
            next_feature = agent.abstractor(torch.FloatTensor(next_channel).to(device))
            next_feature = next_feature.cpu().data.numpy().flatten()
            agent.replay_buffer.add(channel, location, feature, next_channel, next_location, next_feature, action, reward, done)
            state = next_state

            critic_loss, actor_loss = 0, 0
            if episode > 0 or step > warmup:
                critic_loss, actor_loss = agent.train(batch_size, gamma, tau, lr=learning_rate)

            if step % 20 == 0:
                # print("Episode: {}, Step: {}, Total Reward: {}".format(episode, step, episode_reward))
                print("Episode: {}, Step: {}, Critic Loss = {}, Actor Loss = {}".format(episode, step, critic_loss, actor_loss))

            if done:
                break

        if episode > 1 and episode % log_interval == 0:
            agent.save(path)

elif mode == 'plot':
    agent.load(path)
    loc_users = np.array([[5, 5], [5, -5]])
    loc_antennas_init = np.zeros(shape=(num_waveguides,))
    state = env.reset(loc_users=loc_users, loc_antennas=loc_antennas_init)[0]

    for step in range(num_iteration):
        action = agent.select_action(state)
        next_state, _, done, _, _ = env.step(action)

        if done:
            break

        state = next_state

    loc_antennas = env.state["location"]
    sum_rate = env.sum_rate

    plt.figure(figsize=(6, 6))
    plt.xlim(-env.max_range, env.max_range)
    plt.ylim(-env.max_range, env.max_range)

    user_x = loc_users[:, 0]
    user_y = loc_users[:, 1]
    plt.scatter(user_x, user_y, color='blue', label='Users')

    for y_coord in env.loc_waveguides:
        plt.axhline(y=y_coord, color='red', linestyle='--', linewidth=0.8, label=f'y={y_coord}' if y_coord == env.loc_waveguides[0] else "")

    
    plt.scatter(loc_antennas, env.loc_waveguides, color='green', label='Antennas', marker='x')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

else:
    raise NameError("Mode wrong!!!")
