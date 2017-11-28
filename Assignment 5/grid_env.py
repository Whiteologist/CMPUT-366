#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

import numpy as np

# define constants
width = 9  # width of the grid world
height = 6  # height of the grid world
wall = [[7, 0], [7, 1], [7, 2], [2, 1], [2, 2], [2, 3], [5, 4]]  # wall of the grid world

start_state = None
end_state = None
current_state = None


def env_init():
    global start_state, end_state, current_state

    start_state = [0, 2]
    end_state = [8, 0]
    current_state = np.zeros(2)


def env_start():
    """ returns numpy array """
    global current_state

    current_state = [start_state[0], start_state[1]]
    return current_state


def env_step(action):
    global current_state

    # update the current state
    x = current_state[0] + action[0]
    if x < 0:
        x = 0
    if x >= width:
        x = width - 1

    y = current_state[1] + action[1]
    if y < 0:
        y = 0
    if y >= height:
        y = height - 1

    if [x, y] not in wall:
        current_state = [x, y]

    # set up the return value of each action
    reward = 0.0
    is_terminal = False
    if current_state == end_state:
        is_terminal = True
        reward = 1.0

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
