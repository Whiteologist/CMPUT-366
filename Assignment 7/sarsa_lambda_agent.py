#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Sarsa Control agent
           for use on A7 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import *
from tiles3 import *
import numpy as np
import random


# define constants
iht = IHT(4096)
numTilings = 8
sizeOfTilings = [8, 8]
ALPHA = 0.1/numTilings  # step size
LAMBDA = 0.9  # decay rate
EPSILON = 0.0  # exploration rate
initialWeight = [-0.001, 0.0]
GAMMA = 1.0  # discount rate

# initialize variables
W = None  # weight vector
Z = None  # action value vector
last_state = None
last_action = None
last_tile = None  # tiles of last_state


def agent_init():
    global W

    W = np.random.uniform(initialWeight[0], initialWeight[1], (sizeOfTilings[0]+1)*(sizeOfTilings[1]+1)*numTilings*3)


def agent_start(state):
    global last_state, last_action, last_tile, W, Z

    tile = [sizeOfTilings[0]*state[0]/(0.5+1.2), sizeOfTilings[1]*state[1]/(0.07+0.07)]
    Q = np.zeros(3)
    for action in range(3):
        X = np.zeros(len(W))
        for index in tiles(iht, numTilings, tile, [action]):
            X[index] = 1.0
        Q[action] = np.dot(W, X)

    # choose an action
    if rand_un() > EPSILON:
        action = random.choice(random.choice(np.nonzero(Q == np.amax(Q))))
    else:
        action = rand_in_range(3)

    Z = np.zeros(len(W))

    last_state = state
    last_action = action
    last_tile = tile

    return action


def agent_step(reward, state):
    global last_state, last_action, last_tile, W, Z

    for index in tiles(iht, numTilings, last_tile, [last_action]):
        reward -= W[index]
        Z[index] = 1  # replacing traces

    tile = [sizeOfTilings[0]*state[0]/(0.5+1.2), sizeOfTilings[1]*state[1]/(0.07+0.07)]
    Q = np.zeros(3)
    for action in range(3):
        X = np.zeros(len(W))
        for index in tiles(iht, numTilings, tile, [action]):
            X[index] = 1.0
        Q[action] = np.dot(W, X)

    # choose an action
    if rand_un() > EPSILON:
        action = random.choice(random.choice(np.nonzero(Q == np.amax(Q))))
    else:
        action = rand_in_range(3)

    for index in tiles(iht, numTilings, tile, [action]):
        reward += GAMMA * W[index]

    # update weight vector and eligibility trace vector
    W += ALPHA * reward * Z
    Z = GAMMA * LAMBDA * Z

    last_state = state
    last_action = action
    last_tile = tile

    return action


def agent_end(reward):
    global W, Z

    for index in tiles(iht, numTilings, last_tile, [last_action]):
        reward -= W[index]
        Z[index] = 1  # replacing traces

    # update weight vector
    W += ALPHA * reward * Z

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if in_message == 'ValueFunction':
        return [W, iht]
    else:
        return ""
