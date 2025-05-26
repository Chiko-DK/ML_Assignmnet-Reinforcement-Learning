import numpy as np
import matplotlib.pyplot as plt
import random
from FourRooms import FourRooms

# Constants
ACTIONS = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
NUM_EPISODES = 1000
MAX_STEPS = 1000
ALPHA = 0.1
GAMMA = 0.9

def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    return np.argmax(Q.get(state, np.zeros(len(ACTIONS))))

def train_agent():
    env = FourRooms('multi')
    Q = {}
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        env.newEpoch()
        state = (*env.getPosition(), env.getPackagesRemaining())
        total_reward = 0

        epsilon = max(0.01, 1.0 - episode / 600)

        for step in range(MAX_STEPS):
            action = epsilon_greedy(Q, state, epsilon)
            gridType, nextPos, packagesRemaining, isTerminal = env.takeAction(action)
            next_state = (*nextPos, packagesRemaining)

            reward = 100 if isTerminal else -1

            if state not in Q:
                Q[state] = np.zeros(len(ACTIONS))
            if next_state not in Q:
                Q[next_state] = np.zeros(len(ACTIONS))

            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += ALPHA * (reward + GAMMA * Q[next_state][best_next_action] - Q[state][action])

            state = next_state
            total_reward += reward

            if isTerminal:
                break

        episode_rewards.append(total_reward)

    return Q, episode_rewards, env

def run_greedy_policy(Q, env):
    state = (*env.getPosition(), env.getPackagesRemaining())
    for _ in range(MAX_STEPS):
        if state not in Q:
            break
        action = np.argmax(Q[state])
        _, nextPos, packagesRemaining, isTerminal = env.takeAction(action)
        state = (*nextPos, packagesRemaining)
        if isTerminal:
            break

def main():
    print("Training agent for Scenario 2 (multiple packages)...")
    Q, rewards, env = train_agent()

    print("Running greedy policy to show final path...")
    env.newEpoch()
    run_greedy_policy(Q, env)
    env.showPath(-1)



if __name__ == "__main__":
    main()
