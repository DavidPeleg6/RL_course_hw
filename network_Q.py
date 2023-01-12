import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm

# Load environment
env = gym.make('FrozenLake-v1', render_mode='ansi')


def get_one_hot(x, l):
    x = torch.LongTensor([[x]])
    one_hot = torch.FloatTensor(1, l)
    return one_hot.zero_().scatter_(1, x, 1)


num_classes = 4
input_size = 16

model = nn.Sequential(nn.Linear(input_size, num_classes, bias=False))

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Implement Q-Network learning algorithm

# Set learning parameters
GAMMA = .99
epsilon = 0.4
NUM_EPISODES = 4000
NUM_STATES = 16
# create lists to contain total rewards and steps per episode
episodes_lengths_list = []
total_rewards_list = []
for i in tqdm(range(NUM_EPISODES)):
    # Reset environment and get first new observation
    state = env.reset()[0]
    rAll = 0
    terminated, truncated = False, False
    t = 0
    # The Q-Network
    while t < 99:
        t += 1

        Q = model(get_one_hot(state, NUM_STATES))

        action = torch.argmax(Q).item()
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        Q_tag = model(get_one_hot(new_state, NUM_STATES))
        Q_target = Variable(Q.data)
        Q_target[0][action] = reward + torch.mul(GAMMA, torch.max(Q_tag).item())
        Q = model(get_one_hot(state, NUM_STATES))

        loss = criterion(Q_target, Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        state = new_state
        if terminated or truncated:
            # Reduce chance of random action as we train the model.
            epsilon = 1. / ((i / 50) + 10)
            break
    episodes_lengths_list.append(t)
    total_rewards_list.append(rAll)


# A final round to see the network's results in action
model.eval()
for try_count in range(10):
    print(f"new game. try: {try_count+1}")
    state = env.reset()[0]
    terminated, truncated = False, False
    t = 0
    while t < 99:
        t += 1
        Q = model(get_one_hot(state, NUM_STATES))
        action = torch.argmax(Q).item()
        new_state, reward, terminated, truncated, info = env.step(action)
        if new_state != state:
            print(env.render())
        state = new_state
        if terminated or truncated:
            break
    if state == env.observation_space.n - 1:
        break


# Reports
print("Score over time: " + str(sum(total_rewards_list) / NUM_EPISODES))
