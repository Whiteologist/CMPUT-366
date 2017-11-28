#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
from tiles3 import *
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np

iht = IHT(4096)

if __name__ == "__main__":
    # learning curve of Sarsa(lambda) agent
    num_episodes = 200
    num_runs = 50

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r+1
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
            Q = RL_agent_message("ValueFunction")
            W = RL_agent_message("WeightVector")

    fout = open('value', 'w')
    steps = 50
    for i in range(steps):
        for j in range(steps):
            values = []
            for a in range(3):
                X = np.zeros(len(W))
                for index in tiles(iht, steps, [-1.2 + (i * 1.7 / steps), -0.07 + (j * 0.14 / steps)], [a]):
                    X[index] = 1.0
                values.append(-np.dot(W, X))
            height = np.max(values)
            fout.write(repr(height)+'')
        fout.write('\n')
    fout.close()

    print '\nDone!'
