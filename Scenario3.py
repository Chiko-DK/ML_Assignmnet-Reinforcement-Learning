import numpy as np
import random
import argparse
from FourRooms import FourRooms
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 1000
ALPHA = 0.4
GAMMA = 0.95
EPSILON = 0.1
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]

def get_state(pos, packages_left):
    return (pos[0], pos[1], packages_left)

def choose_action(Q, state, epsilon):
    if random.random() < epsilon or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic', '-s', action='store_true', help='Use stochastic actions')
    args = parser.parse_args()

    fourRoomObj = FourRooms(scenario='rgb', stochastic=args.stochastic)
    if args.stochastic:
        print("Stochastic actions enabled.")
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
            _, new_pos, packages_left, is_terminal = fourRoomObj.takeAction(action)
            next_state = get_state(new_pos, packages_left)

            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in ACTIONS}

            # If the agent incorrectly picks a package, it ends early
            reward = 10 if packages_left < state[2] else -1
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
