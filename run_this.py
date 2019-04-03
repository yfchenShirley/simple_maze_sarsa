"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       walls       [reward = -5].
Yellow bin circle:      exit    [reward = +100].
All other states:       ground      [reward = -1].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from Sarsa_brain import SarsaTable

EPIS = 1000
plot_x = []

def update():
    for episode in range(EPIS):
        
        # initial observation
        observation = env.reset()
        reward_total = 0

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            reward_total += reward

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                plot_x.append(reward_total)
                print(f"Episode {episode}:")
                print(f"Total rewards: {reward_total}")
                break

    # end of game
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    # Data for plotting
   
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(plot_x, range(EPIS))

    ax.set(xlabel='episode (s)', ylabel='Total Rewards',
           title='Total rewards at each episode')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()