import numpy as np
import torch
import gym
import argparse

from gym.wrappers import FlattenObservation
from itertools import count
from my_td3_networks import TD3

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="PinchingAntenna-v1")  # gym environment
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=100, type=int)  # replay buffer size
parser.add_argument('--num_iteration', default=300, type=int)  # num of games
parser.add_argument('--batch_size', default=10, type=int)  # mini batch size
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
parser.add_argument('--policy_noise', default=0.1, type=float)
parser.add_argument('--noise_clip', default=0.1, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int)
parser.add_argument('--print_log', default=5, type=int)
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
        state = env.reset()
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            ep_r += reward
            env.render()
            if done or t == 2000:
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                break
            state = next_state

elif args.mode == 'train':
    print("====================================")
    print("Collection Experience...")
    print("====================================")
    if args.load:
        agent.load()

    for i in range(args.num_iteration):
        state = env.reset()[0]

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
                print('Episode {},  the memory size is {} '.format(t + 1, len(agent.memory.storage)))
            if len(agent.memory.storage) >= args.capacity - 1:
                agent.update(10)

            state = next_state
            if done or t == args.max_episode - 1:
                agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                if i % args.print_log == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                ep_r = 0
                break

        if i % args.log_interval == 0:
            agent.save()

else:
    raise NameError("mode wrong!!!")
