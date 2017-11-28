#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the random walk's problem environment
  and the semi-gradient TD(0) agents using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
from rndmwalk_policy_evaluation import *  # true value function
import math
RLGlue("randomWalk_env", "randomWalk_agent")

if __name__ == "__main__":

    num_of_runs = 10
    run = 0

    x_axis = np.zeros(5000)
    y_axis = np.delete(compute_value_function(), 0)
    RMSE_tabular = np.zeros(5000)
    RMSE_tile = np.zeros(5000)
    RMSE_aggregation = np.zeros(5000)
    for i in range(5000):
        x_axis[i] = i+1

    RL_agent_message("gamma:1.0")
    while run < num_of_runs:
        run += 1

        # agent 1: Tabular Encoding
        print("Tabular Encoding: Run " + str(run))
        print("\n")
        RL_agent_message("agent:Tabular Encoding")
        RL_agent_message("alpha:0.5")
        RL_init()
        num_episodes = 0
        while num_episodes < 5000:
            RL_episode(10000)
            num_episodes = RL_num_episodes()
            value = RL_agent_message('ValueFunction')
            RMSE_tabular[num_episodes-1] += math.sqrt(np.sum((y_axis - value)*(y_axis - value))/1000.0)

        # agent 2: Tile Coding
        print("Tile Coding: Run " + str(run))
        print("\n")
        RL_agent_message("agent:Tile Coding")
        RL_agent_message("num of tiling:50")
        RL_agent_message("tile width:0.2")
        RL_agent_message("alpha:0.01")
        RL_init()
        num_episodes = 0
        while num_episodes < 5000:
            RL_episode(10000)
            num_episodes = RL_num_episodes()
            value = RL_agent_message('ValueFunction')
            RMSE_tile[num_episodes-1] += math.sqrt(np.sum((y_axis - value)*(y_axis - value))/1000.0)

        # agent 3: State Aggregation
        print("State Aggregation: Run " + str(run))
        print("\n")
        RL_agent_message("agent:Tile Coding")
        RL_agent_message("num of tiling:1")
        RL_agent_message("tile width:0.1")
        RL_agent_message("alpha:0.1")
        RL_init()
        num_episodes = 0
        while num_episodes < 5000:
            RL_episode(10000)
            num_episodes = RL_num_episodes()
            value = RL_agent_message('ValueFunction')
            RMSE_aggregation[num_episodes-1] += math.sqrt(np.sum((y_axis - value)*(y_axis - value))/1000.0)

    print "Done!"

    plt.plot(x_axis, RMSE_tabular/num_of_runs, label='Tabular Encoding')
    plt.plot(x_axis, RMSE_tile/num_of_runs, label='Tile Coding')
    plt.plot(x_axis, RMSE_aggregation/num_of_runs, label='State Aggregation')
    plt.xlim([1, 5000])
    plt.xlabel('Episodes')
    plt.ylabel('RMSVE')
    plt.legend()
    plt.show()

    RL_cleanup()
