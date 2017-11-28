#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

import numpy as np

# define constants
width = 10  # width of the grid world
height = 7  # height of the grid world
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # the wind strength of each column

start_state = None
end_state = None
current_state = None


def env_init():
    global start_state, end_state, current_state

    start_state = [0, 3]
    end_state = [7, 3]
    current_state = np.zeros(2)


def env_start():
    """ returns numpy array """
    global current_state

    current_state = [start_state[0], start_state[1]]
    return current_state


def env_step(action):
    global current_state

    x = current_state[0] + action[0]
    if x < 0:
        x = 0
    if x >= width:
        x = width - 1

    y = current_state[1] + action[1] - wind[x]
    if y < 0:
        y = 0
    if y >= height:
        y = height - 1

    # update the current state
    current_state = [x, y]

    # set up the return value of each action
    reward = -1.0
    is_terminal = False
    if current_state == end_state:
        is_terminal = True
        reward = 0.0

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result


def env_cleanup():
    #
    return


def env_message(in_message):  # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
