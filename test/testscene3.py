# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from FourRooms import FourRooms

# # Hyperparameters
# EPISODES = 1000
# ALPHA = 0.2
# GAMMA = 0.95
# EPSILON_START = 1.0
# EPSILON_END = 0.05
# DECAY_RATE = 0.995

# ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# # Convert current state into a unique hashable key for Q-table
# def get_state(pos, rgb_progress):
#     return (pos[0], pos[1], tuple(rgb_progress))

# def epsilon_greedy(Q, state, epsilon):
#     if random.random() < epsilon or state not in Q:
#         return random.choice(ACTIONS)
#     return max(Q[state], key=Q[state].get)

# def update_q(Q, state, action, reward, next_state, alpha, gamma):
#     if state not in Q:
#         Q[state] = {a: 0.0 for a in ACTIONS}
#     if next_state not in Q:
#         Q[next_state] = {a: 0.0 for a in ACTIONS}

#     max_q_next = max(Q[next_state].values())
#     Q[state][action] += alpha * (reward + gamma * max_q_next - Q[state][action])

# def train():
#     env = FourRooms(scenario='rgb', stochastic=False)
#     Q = {}
#     rewards = []
#     epsilon = EPSILON_START

#     for episode in range(EPISODES):
#         env.newEpoch()
#         total_reward = 0
#         rgb_collected = [False, False, False]  # R, G, B
#         state = get_state(env.getPosition(), rgb_collected)

#         while not env.isTerminal():
#             action = epsilon_greedy(Q, state, epsilon)
#             reward, new_pos, rgb_status, is_terminal = env.takeAction(action)
#             next_state = get_state(new_pos, rgb_status)

#             # Apply reward logic
#             if is_terminal:
#                 reward = 10 if rgb_status == [True, True, True] else -10
#             else:
#                 reward = -1

#             update_q(Q, state, action, reward, next_state, ALPHA, GAMMA)

#             state = next_state
#             total_reward += reward

#         epsilon = max(EPSILON_END, epsilon * DECAY_RATE)
#         rewards.append(total_reward)

#         if (episode + 1) % 100 == 0:
#             print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}")

#     return Q, rewards, env

# def test(Q, env):
#     env.newEpoch()
#     rgb_collected = [False, False, False]
#     state = get_state(env.getPosition(), rgb_collected)

#     while not env.isTerminal():
#         if state not in Q:
#             break
#         action = max(Q[state], key=Q[state].get)
#         _, new_pos, rgb_status, _ = env.takeAction(action)
#         state = get_state(new_pos, rgb_status)

#     env.showPath(-1)


# def main():
#     print("Training agent for Scenario 3 (ordered RGB)...")
#     Q, rewards, env = train()
#     print("Training complete. Showing final path with greedy policy...")
#     test(Q, env)

# if __name__ == '__main__':
#     main()

import numpy as np
import random
from FourRooms import FourRooms
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.2
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# Treat RGB progress as a single integer state
def get_state(pos, rgb_progress):
    return (pos[0], pos[1], rgb_progress)

def choose_action(Q, state, epsilon):
    if random.random() < epsilon or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

def main():
    fourRoomObj = FourRooms(scenario='rgb', stochastic=False)
    Q = {}
    rewards = []

    for episode in range(EPISODES):
        fourRoomObj.newEpoch()
        total_reward = 0
        state = get_state(fourRoomObj.getPosition(), fourRoomObj.getPackagesRemaining())

        while not fourRoomObj.isTerminal():
            if state not in Q:
                Q[state] = {a: 0.0 for a in ACTIONS}

            action = choose_action(Q, state, EPSILON)
            reward, new_pos, new_rgb_status, is_terminal = fourRoomObj.takeAction(action)
            next_state = get_state(new_pos, new_rgb_status)

            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in ACTIONS}

            # If agent collects R, G, B in order it gets +10; wrong order ends episode early with penalty
            if is_terminal:
                if new_rgb_status == 3:  # Assuming 3 = all packages collected in correct order
                    reward = 10
                else:
                    reward = -10
            else:
                reward = -1

            best_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total reward = {total_reward}")

    print("Training complete. Showing final path...")
    fourRoomObj.showPath(-1)

if __name__ == '__main__':
    main()
