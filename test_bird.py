import sys
from collections import defaultdict

import numpy as np
import time as t

import cv2

from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)

p.init()
reward = 0.0

def run_episodes(p):
        episode = []
        state_infos = ['player_y', 'player_vel', 'next_pipe_dist_to_player',
                        'next_pipe_top_y', 'next_pipe_bottom_y']
        p.reset_game()
        state = []
        current_state = p.getGameState()
        print("\n Current state:", current_state)
        for info in state_infos:
                state.append(current_state.get(info))
        while True:
                next_state = []
                action = p.getActionSet()
                action = action[0] if (state[0] > state[3] and state[2] > 140.0) else action[1]
                print(p.score())
                reward = p.act(action)
                t.sleep(0.1)
                if action:
                        action = 1
                else:
                        action = 0
                next_state_info = p.getGameState()
                for info in state_infos:
                        next_state.append(next_state_info.get(info))
                episode.append((tuple(state), action, reward))
                state = next_state
                if p.game_over():
                        break

        print(reward)
        return episode



def update_values(episode, Q, return_values, N, gamma=1.0):

        for s, a, r in episode:
                first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s)
                G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
                # print("\n Value G: ", G, s, a, return_values)
                return_values[s][a] += G
                N[s][a] += 1.0
                Q[s][a] = return_values[s][a] / N[s][a]

        return Q

def mc_prediction(p, num_episodes, gamma=1.0):
        returns_sum = (defaultdict(lambda: np.zeros(2)))
        N = defaultdict(lambda: np.zeros(2))
        Q = defaultdict(lambda: np.zeros(2))
        returns_sum[tuple([1, 2, 4])][0] = 9.0
        # print("return", returns_sum)

        for i_episode in range(1, num_episodes+1):
                if i_episode % 1000 == 0:
                        print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                        sys.stdout.flush()

                episode = run_episodes(p)

                Q = update_values(episode, Q, returns_sum, N)

        return Q



Q = []

Q = mc_prediction(p, 2)

print("Q values: ", Q)

def test_value(p, Q):
        episode = []
        state_infos = ['player_y', 'player_vel', 'next_pipe_dist_to_player',
                        'next_pipe_top_y', 'next_pipe_bottom_y']
        p.reset_game()
        state = []
        current_state = p.getGameState()
        print("\n Current state:", current_state)
        for info in state_infos:
                state.append(current_state.get(info))
        while True:
                next_state = []
                action = p.getActionSet()
                action = action[0] if (state[0] > state[3] and state[2] > 140.0) else action[1]
                print(p.getGameState())
                reward = p.act(action)
                t.sleep(0.1)
                if action:
                        action = 1
                else:
                        action = 0
                next_state_info = p.getGameState()
                for info in state_infos:
                        next_state.append(next_state_info.get(info))
                episode.append((tuple(state), action, reward))
                state = next_state
                if p.game_over():
                        break

# for i in range(5):
#    if p.game_over():
#            p.reset_game()

#    observation = p.getScreenRGB()
#    states = p.getGameState()
#    action = np.random.choice(p.getActionSet())
#    reward = p.act(action)
#    print("\nScore: ", states)
#    t.sleep(0.01)