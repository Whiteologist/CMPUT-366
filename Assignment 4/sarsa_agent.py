#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import *
import numpy as np

# define constants
width = 10  # width of the grid world
height = 7  # height of the grid world
moves = 8  # number of actions possible
actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

alpha = 0.5  # step size
epsilon = 0.1  # probability of exploration

Q = None
last_state = None
last_action = None


def agent_init():
    global Q

    Q = np.full((width, height, moves), -1.0)
    Q[7, 3, :] = 0.0

    if moves in [4, 8, 9]:
        if moves >= 8:
            for move in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
                actions.append(move)
        if moves == 9:
            actions.append([0, 0])
    else:
        print("Invalid # of actions!")
        exit(0)


def agent_start(state):
    # choose an action based on epsilon-greedy algorithm
    global last_state, last_action

    # choose an action based on epsilon-greedy algorithm
    if rand_un() > epsilon:
        current_action = np.argmax(Q[state[0], state[1]])
    else:
        current_action = rand_in_range(moves)

    last_state = [state[0], state[1]]
    last_action = current_action

    return actions[current_action]


def agent_step(reward, state):
    # set up the destination of each action
    global last_state, last_action

    # choose an action based on epsilon-greedy algorithm
    if rand_un() > epsilon:
        current_action = np.argmax(Q[state[0], state[1]])
    else:
        current_action = rand_in_range(moves)

    Q[last_state[0], last_state[1], last_action] += alpha * (reward + Q[state[0], state[1], current_action] - Q[last_state[0], last_state[1], last_action])

    last_state = [state[0], state[1]]
    last_action = current_action

    return actions[current_action]


def agent_end(reward):
    global Q, last_state, last_action

    Q[last_state[0], last_state[1], last_action] += alpha * (reward - Q[last_state[0], last_state[1], last_action])

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if in_message == 'ValueFunction':
        return
    else:
        return "I don't know what to return!!"
