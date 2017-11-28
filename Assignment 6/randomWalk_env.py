#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the random walk's problem environment
  and the semi-gradient TD(0) agents using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""
current_state = None


def env_init():
    return


def env_start():
    global current_state

    current_state = 500
    return current_state


def env_step(action):
    global current_state

    # update the current state
    current_state += action

    # set up the return value of each action
    reward = 0.0
    is_terminal = False
    if current_state < 1:
        is_terminal = True
        reward = -1.0
    elif current_state > 1000:
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
    in_message : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
