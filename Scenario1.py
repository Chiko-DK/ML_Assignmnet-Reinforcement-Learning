import numpy as np
import random
from FourRooms import FourRooms

# Hyperparameters
EPISODES = 500
ALPHA = 0.1      # Learning rate
GAMMA = 0.95     # Discount factor
EPSILON = 0.1    # Exploration rate

ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

# Returns a state representation
def get_state(pos, packages_left):
    return (pos[0], pos[1], packages_left)

# ε-greedy action selection
def choose_action(Q, state):
    if random.random() < EPSILON or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

# Initialize fourRoomsObjironment
fourRoomsObj = FourRooms(scenario='simple', stochastic=False)
print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))

Q = {}  # Q-table

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
fourRoomsObj.showPath(-1)
