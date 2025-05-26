# import numpy as np
# import random
# import argparse
# from FourRooms import FourRooms
# import matplotlib.pyplot as plt

# # Define Q-learning hyperparameters
# GAMMA = 0.95         # Discount factor
# ALPHA = 0.1          # Learning rate

# ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# def epsilon_greedy_action(Q, state, epsilon):
#     if random.random() < epsilon:
#         return random.choice(ACTIONS)
#     if state in Q:
#         return max(Q[state], key=Q[state].get)
#     return random.choice(ACTIONS)

# def get_state(position, packages_remaining):
#     return (position[0], position[1], packages_remaining)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
#     parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
#     parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
#     parser.add_argument('--plot', action='store_true', help='Plot cumulative rewards')
#     args = parser.parse_args()

#     env = FourRooms(scenario='simple', stochastic=args.stochastic)
#     Q = {}

#     episode_rewards = []

#     for ep in range(args.episodes):
#         env.newEpoch()
#         total_reward = 0
#         state = get_state(env.getPosition(), env.getPackagesRemaining())

#         while not env.isTerminal():
#             if state not in Q:
#                 Q[state] = {a: 0.0 for a in ACTIONS}

#             action = epsilon_greedy_action(Q, state, args.epsilon)
#             _, new_pos, packages_left, terminal = env.takeAction(action)
#             next_state = get_state(new_pos, packages_left)

#             if next_state not in Q:
#                 Q[next_state] = {a: 0.0 for a in ACTIONS}

#             reward = 10 if terminal else -1
#             best_next_q = max(Q[next_state].values())

#             # Q-learning update rule
#             Q[state][action] += ALPHA * (reward + GAMMA * best_next_q - Q[state][action])

#             total_reward += reward
#             state = next_state

#         episode_rewards.append(total_reward)
#         if (ep + 1) % 100 == 0:
#             print(f"Episode {ep+1}: Total Reward = {total_reward}")

#     print("Training complete. Showing final path...")
#     env.showPath(-1)

#     if args.plot:
#         plt.plot(episode_rewards)
#         plt.title(f"Episode Rewards (ε={args.epsilon})")
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward")
#         plt.grid(True)
#         plt.savefig(f"rewards_epsilon_{args.epsilon}.png")
#         plt.show()

# if __name__ == '__main__':
#     main()

import numpy as np
import random
from FourRooms import FourRooms
import matplotlib.pyplot as plt

EPISODES = 500
ALPHA = 0.1
GAMMA = 0.95
EPSILONS = [0.1, 0.3]  # Two exploration strategies
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

def get_state(pos, packages_left):
    return (pos[0], pos[1], packages_left)

def choose_action(Q, state, epsilon):
    if random.random() < epsilon or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

def train_agent(epsilon):
    env = FourRooms(scenario='simple', stochastic=False)
    Q = {}
    rewards = []

    for episode in range(EPISODES):
        env.newEpoch()
        total_reward = 0
        state = get_state(env.getPosition(), env.getPackagesRemaining())

        while not env.isTerminal():
            if state not in Q:
                Q[state] = {a: 0.0 for a in ACTIONS}

            action = choose_action(Q, state, epsilon)
            _, new_pos, packages_left, is_terminal = env.takeAction(action)
            next_state = get_state(new_pos, packages_left)

            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in ACTIONS}

            reward = 10 if is_terminal else -1
            best_next = max(Q[next_state].values())

            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"ε={epsilon} | Episode {episode+1}: Total reward = {total_reward}")

    # Save the final path image
    env.showPath(-1, savefig=f"path_epsilon_{epsilon}.png")

    return rewards

# Run training for both ε values
results = {}
for eps in EPSILONS:
    results[eps] = train_agent(eps)

# Plot rewards
plt.figure(figsize=(10, 5))
for eps in EPSILONS:
    plt.plot(results[eps], label=f"ε={eps}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode for Different ε-Greedy Strategies")
plt.legend()
plt.grid(True)
plt.savefig("epsilon_comparison_plot.png")
plt.show()
