import numpy as np
import random
import argparse
from FourRooms import FourRooms

# Hyperparameters
EPISODES = 500
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1

ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

def get_state(pos, packages_left):
    return (pos[0], pos[1], packages_left)

# ε-greedy action selection
def choose_action(Q, state):
    if random.random() < EPSILON or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

parser = argparse.ArgumentParser()
parser.add_argument('--stochastic', '-s', action='store_true', help='Use stochastic actions')
args = parser.parse_args()

# Initialize fourRoomsObject
fourRoomsObj = FourRooms(scenario='simple', stochastic=args.stochastic)
if args.stochastic:
    print("Stochastic actions enabled.")

Q = {}  # Q-value

for episode in range(EPISODES):
    fourRoomsObj.newEpoch()
    total_reward = 0
    state = get_state(fourRoomsObj.getPosition(), fourRoomsObj.getPackagesRemaining())

    while not fourRoomsObj.isTerminal():
        if state not in Q:
            Q[state] = {a: 0.0 for a in ACTIONS}

        action = choose_action(Q, state)
        _, new_pos, packages_left, is_terminal = fourRoomsObj.takeAction(action)

        next_state = get_state(new_pos, packages_left)
        if next_state not in Q:
            Q[next_state] = {a: 0.0 for a in ACTIONS}

        reward = 10 if is_terminal else -1
        best_next = max(Q[next_state].values())

        Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

        state = next_state
        total_reward += reward

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Show final path
print("Training complete. Showing final path...")
fourRoomsObj.showPath(-1)
