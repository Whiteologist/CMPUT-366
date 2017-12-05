#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

import numpy as np
from rl_glue import *  # Required for RL-Glue
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
RLGlue("mountaincar", "sarsa_lambda_agent")

if __name__ == "__main__":
    # learning curve of Sarsa(lambda) agent
    num_episodes = 200
    num_runs = 50

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r+1
        RL_agent_message("LearningCurve")
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps[r,e] = RL_num_steps()
    np.save('steps',steps)

    # 3D-plot of state-values
    num_episodes = 1000
    num_runs = 1

    for r in range(num_runs):
        print "Computing state-values..."
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fout = open('value', 'w')
    steps = 50
    x = np.arange(-1.2, 0.5, 1.7 / steps)
    y = np.arange(-0.07, 0.07, 0.14 / steps)
    x, y = np.meshgrid(x, y)
    Q = RL_agent_message("ValueFunction")
    for i in range(steps):
        for j in range(steps):
            height = Q[j][i]
            fout.write(repr(height)+'')
        fout.write('\n')
    fout.close()

    ax.set_xticks([-1.2, 0.5])
    ax.set_yticks([-0.07, 0.07])
    ax.set_zticks([0, np.amax(Q)])
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.plot_surface(x, y, Q)
    plt.show()

    print '\nDone!'
