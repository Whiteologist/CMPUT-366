#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Dyna-Q Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import *
import numpy as np
import random

# define grid world
width = 9  # width of the grid world
height = 6  # height of the grid world
actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

# define constants
alpha = None  # step size (defined in experiment)
epsilon = None  # probability of exploration (defined in experiment)
gamma = 0.95  # discount rate
n = None  # planning steps (defined in experiment)

# initialize variables
Q = None
Model = None
last_state = None
last_action = None
observed_states = None
previous_actions = None


def agent_init():
    global Q, Model, observed_states, previous_actions

    Q = np.zeros((width, height, 4))
    Q[8, 0, :] = 1.0
    Model = []
    for i in range(width):
        row = []
        for j in range(height):
            column = []
            for k in range(4):
                column.append([])
            row.append(column)
        Model.append(row)
    observed_states = []
    previous_actions = np.zeros((width, height, 4))


def agent_start(state):
    # choose an action based on epsilon-greedy algorithm
    global last_state, last_action, observed_states, previous_actions, epsilon

    # choose an action based on epsilon-greedy algorithm
    if rand_un() > epsilon:
        current_action = random.choice(random.choice(np.nonzero(Q[state[0], state[1]] == np.amax(Q[state[0], state[1]]))))
    else:
        current_action = rand_in_range(4)

    if state not in observed_states:
        observed_states.append(state)
    previous_actions[state[0], state[1], current_action] = 1.0

    last_state = [state[0], state[1]]
    last_action = current_action

    return actions[current_action]


def agent_step(reward, state):
    # set up the destination of each action
    global last_state, last_action, alpha, epsilon, n

    # choose an action based on epsilon-greedy algorithm
    if rand_un() > epsilon:
        current_action = random.choice(random.choice(np.nonzero(Q[state[0], state[1]] == np.amax(Q[state[0], state[1]]))))
    else:
        current_action = rand_in_range(4)

    Q[last_state[0], last_state[1], last_action] += alpha * (reward + gamma * max(Q[state[0], state[1]]) - Q[last_state[0], last_state[1], last_action])
    Model[last_state[0]][last_state[1]][last_action] = [reward, state[0], state[1]]

    for i in range(n):
        s = random.choice(observed_states)
        choice = []
        for action in range(4):
            if previous_actions[s[0], s[1], action] == 1.0:
                choice.append(action)
        a = random.choice(choice)
        r = Model[s[0]][s[1]][a][0]
        x = Model[s[0]][s[1]][a][1]
        y = Model[s[0]][s[1]][a][2]
        Q[s[0], s[1], a] += alpha * (r + gamma * max(Q[x, y]) - Q[s[0], s[1], a])

    last_state = [state[0], state[1]]
    last_action = current_action

    if state != [8, 0]:
        if state not in observed_states:
            observed_states.append(state)
        previous_actions[state[0], state[1], current_action] = 1.0

    return actions[current_action]


def agent_end(reward):
    global Q, last_state, last_action

    Q[last_state[0], last_state[1], last_action] += alpha * (reward - Q[last_state[0], last_state[1], last_action])
    Model[last_state[0]][last_state[1]][last_action] = [reward, 8, 0]

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global alpha, epsilon, n
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    try:
        line = in_message.split(",")
        alpha = float(line[0])  # step size
        epsilon = float(line[1])  # probability of exploration
        n = int(line[2])  # planning steps
        return
    except ValueError:
        return "I don't know what to return!!"
