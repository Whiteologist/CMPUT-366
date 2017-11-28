#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Windy Gridworld's problem environment
  and the SARSA agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windy_env", "sarsa_agent")


if __name__ == "__main__":
    num_episodes = 0
    max_steps = 8000
    total_step = 0
    output = open("output.txt", "w")

    RL_init()
    while total_step < 8000:
        RL_episode(max_steps)
        total_step += RL_num_steps()
        output.write("%d,%d\n" %(total_step, num_episodes))
        print(total_step, num_episodes)
        num_episodes = RL_num_episodes()

    RL_cleanup()
    output.close()

    print "Done!"
