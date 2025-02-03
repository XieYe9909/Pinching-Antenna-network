import numpy as np
import torch
import gym
import argparse
import matplotlib.pyplot as plt

from gym.wrappers import FlattenObservation
from itertools import count
from my_td3_networks import TD3

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="PinchingAntenna-v1")  # gym environment
parser.add_argument('--mode', default='plot', type=str)  # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=50, type=int)  # replay buffer size
parser.add_argument('--num_iteration', default=300, type=int)  # num of games
parser.add_argument('--batch_size', default=5, type=int)  # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--max_action', default=1, type=float)
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.01, type=float)
parser.add_argument('--noise_clip', default=0.01, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.01, type=float)
parser.add_argument('--max_episode', default=1000, type=int)
parser.add_argument('--print_log', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'data/'

num_waveguides = 2
num_users = 2

env = gym.make(id=args.env_name, num_waveguides=num_waveguides, num_users=num_users)
if isinstance(env.observation_space, gym.spaces.Dict):
    env = FlattenObservation(env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3(state_dim, action_dim, args, path, device)
ep_r = 0

if args.mode == 'test':
    agent.load()
    for i in range(args.iteration):
        state = env.reset()[0]
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            ep_r += reward
            env.render()
            if done or t == args.max_episode - 1:
                print("Ep_i is {}, the ep_r is {:0.2f}, the step is {}".format(i, ep_r, t))
                break
            state = next_state

elif args.mode == 'train':
    print("====================================")
    print("Collection Experience...")
    print("====================================")
    if args.load:
        agent.load()

    for i in range(args.num_iteration):
        loc_antennas_init = np.zeros(shape=(num_waveguides,))
        state = env.reset(loc_antennas=loc_antennas_init)[0]
        ep_r = env.sum_rate
        agent.memory.clear()

        for t in range(args.max_episode):
            action = agent.select_action(state)
            action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
            action = action.clip(-args.max_action, args.max_action)
            next_state, reward, done, _, _ = env.step(action)
            ep_r += reward
            if args.render and i >= args.render_interval:
                env.render()

            agent.memory.push((state, next_state, action, reward, float(done)))
            if (t + 1) % 20 == 0:
                print('Episode is {},  the memory size is {} '.format(t + 1, len(agent.memory.storage)))
            if len(agent.memory.storage) >= args.capacity:
                agent.update(10)

            state = next_state
            if done or t == args.max_episode - 1:
                agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                if i % args.print_log == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                break

        if i % args.log_interval == 0:
            agent.save()

elif args.mode == 'plot':
    agent.load()
    loc_users = np.array([[5, 5], [5, -5]])
    loc_antennas_init = np.zeros(shape=(num_waveguides,))
    state = env.reset(loc_users=loc_users, loc_antennas=loc_antennas_init)[0]

    for t in count():
        action = agent.select_action(state)
        next_state, _, done, _, _ = env.step(action)
        env.render()
        if done or t == args.max_episode - 1:
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
    raise NameError("mode wrong!!!")
