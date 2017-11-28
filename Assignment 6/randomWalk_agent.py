#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for semi-gradient TD(0) Agent
           for use on A6 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import *
from tiles3 import *
import numpy as np
import random

iht = IHT(1024)

# define constants
agent = None
num_of_tiling = None
tile_width = None
alpha = None  # step size
gamma = None  # discount parameter

# initialize variables
V = None   # value function
W = None   # weight vector
X = None   # feature vector
last_state = None
last_tile = None  # tiles of last_state
current_tile = None  # tiles of current_state
last_X = None  # feature vector of last_tile
current_X = None   # feature vector of current_tile


def agent_init():
    global V, W, X

    if agent == "Tabular Encoding":
        W = np.zeros(1000)
        X = np.identity(1000)
    elif agent == "Tile Coding":
        V = np.zeros(1000)
        W = np.zeros(1000+int(1000*tile_width))
    elif agent == "State Aggregation":
        W = np.zeros(10)
        X = np.identity(10)


def agent_start(state):
    global last_state, last_tile, current_tile

    # choose an action
    action = random.choice([-(rand_in_range(100)+1), rand_in_range(100)+1])

    last_state = state
    if agent == "Tile Coding":
        current_tile = float(state/(1000*tile_width))
        last_tile = current_tile

    return action


def agent_step(reward, state):
    global last_state, last_tile, current_tile, last_X, current_X, V, W, X

    # choose an action
    action = random.choice([-(rand_in_range(100)+1), rand_in_range(100)+1])

    if agent == "Tabular Encoding":
        # update weight vector
        W += alpha * (reward + gamma * np.dot(W, X[state-1]) - np.dot(W, X[last_state-1])) * X[last_state-1]

    elif agent == "Tile Coding":
        last_X = np.zeros(len(W))
        for index in tiles(iht, num_of_tiling, [last_tile]):
            last_X[index] = 1.0

        current_X = np.zeros(len(W))
        current_tile = float(state/(1000*tile_width))
        for index in tiles(iht, num_of_tiling, [current_tile]):
            current_X[index] = 1.0

        # update weight vector
        W += alpha * (reward + gamma * np.dot(W, current_X) - np.dot(W, last_X)) * last_X

        # update value function
        V[last_state-1] = np.dot(W, last_X)

        last_tile = current_tile

    elif agent == "State Aggregation":
        # update weight vector
        W += alpha * (reward + gamma * np.dot(W, X[(state-1)/100]) - np.dot(W, X[(last_state-1)/100])) * X[(last_state-1)/100]

    last_state = state

    return action


def agent_end(reward):
    global last_state, last_tile, current_tile, last_X, V, W, X

    if agent == "Tabular Encoding":
        # update weight vector
        W += alpha * (reward - np.dot(W, X[last_state-1])) * X[last_state-1]

    elif agent == "Tile Coding":
        last_X = np.zeros(len(W))
        for index in tiles(iht, num_of_tiling, [last_tile]):
            last_X[index] = 1.0

        # update weight vector
        W += alpha * (reward - np.dot(W, last_X)) * last_X

        # update value function
        V[last_state-1] = np.dot(W, last_X)

    elif agent == "State Aggregation":
        # update weight vector
        W += alpha * (reward - np.dot(W, X[(last_state-1)/100])) * X[(last_state-1)/100]

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global agent, alpha, gamma, num_of_tiling, tile_width, V, W
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    try:
        line = in_message.split(':')

        if line[0] == 'agent':
            agent = line[1]
        if line[0] == 'alpha':
            if agent == 'Tile Coding':
                alpha = float(line[1])/num_of_tiling
            else:
                alpha = float(line[1])
        if line[0] == 'gamma':
            gamma = float(line[1])
        if line[0] == 'num of tiling':
            num_of_tiling = int(line[1])
        if line[0] == 'tile width':
            tile_width = float(line[1])
        if line[0] == 'ValueFunction':
            if agent == 'Tile Coding':
                return V
            else:
                if agent == 'State Aggregation':
                    value = np.zeros(1000)
                    for i in range(1000):
                        value[i] = W[i/100]
                    return value
                return W

        return

    except IndexError:
        return "I don't know what to return!!"
    except TypeError:
        return "I don't know what to return!!"
    except ValueError:
        return "I don't know what to return!!"
