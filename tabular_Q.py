import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm

# Load environment
env = gym.make('FrozenLake-v1', render_mode="ansi")

# Implement Q-Table learning algorithm
# Initialize table with all zeros

# Set learning parameters
LEARNING_RATE = .8
GAMMA = .95
NUM_EPISODES = 4000

Q = np.zeros([env.observation_space.n, env.action_space.n])
rewards_list = []
for episode_number in tqdm(range(NUM_EPISODES)):
    # Reset environment and get first new observation
    state = env.reset()[0]
    total_episode_reward = 0  # Total reward during current episode
    terminated = False
    t = 0
    # The Q-Table learning algorithm
    while t < 99:
        t += 1
        action = np.argmax(Q[state] + np.random.normal(0, 0.01, size=Q.shape[1]))
        next_state, reward, terminated, truncated, info = env.step(action)
        CAPITAL_GAMMA = reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
        Q[state][action] = Q[state][action] + LEARNING_RATE * CAPITAL_GAMMA
        total_episode_reward += reward
        if terminated or truncated:
            break
        state = next_state

    rewards_list.append(total_episode_reward)

# a Final round to show the actual results
for try_count in range(10):
    print(f"new game. try: {try_count+1}")
    state = env.reset()[0]
    terminated, truncated = False, False
    t = 0
    while t < 99:
        t += 1
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        if next_state != state:
            print(env.render())
        state = next_state
        if terminated or truncated:
            break
    if state == env.observation_space.n - 1:
        break


# Reports
print("Score over time: " + str(sum(rewards_list) / NUM_EPISODES))
print("Final Q-Table Values")
print(Q)
