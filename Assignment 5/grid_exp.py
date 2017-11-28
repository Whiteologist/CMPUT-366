#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gridworld's problem environment
  and the Dyna-Q agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

import numpy as np
from rl_glue import *  # Required for RL-Glue
RLGlue("grid_env", "Dyna-Q_agent")


if __name__ == "__main__":
    output = open("output1.txt", "w")
    planning_steps = [0, 5, 50]
    np.random.seed(1)
    for step in planning_steps:
        data = np.zeros(50)
        RL_agent_message("0.1, 0.1, " + str(step))
        print "\nplanning step:", step
        for run in range(10):
            RL_init()
            print "\n\trun number: ", run + 1
            num_episodes = 0
            while num_episodes < 50:
                RL_episode(10000)
                data[num_episodes] += RL_num_steps()
                num_episodes = RL_num_episodes()

        for episode in range(50):
            output.write("%d,%d\n" % (episode + 1, data[episode]/10))

        RL_cleanup()
    output.close()

    print "\nDone!"
