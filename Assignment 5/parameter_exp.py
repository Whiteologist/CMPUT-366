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
    output = open("output2.txt", "w")
    step_sizes = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    epsilon = 0.06
    np.random.seed(2)
    for alpha in step_sizes:
        average_step = 0
        RL_agent_message(str(alpha) + ", " + str(epsilon) + ", 5")
        print "\nstep size:", alpha
        for run in range(10):
            RL_init()
            print "\n\trun number: ", run + 1
            num_episodes = 0
            while num_episodes < 50:
                RL_episode(10000)
                average_step += RL_num_steps()
                num_episodes = RL_num_episodes()

        output.write("%f,%d\n" % (alpha, average_step/500))

        RL_cleanup()
    output.close()

    print "\nDone!"
