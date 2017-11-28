#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import *
import numpy as np
import pickle
import random


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    # initialize the policy array in a smart way
    global Q, pi, return_track, return_value, episode

    Q = np.zeros((100, 100))
    pi = np.zeros(100)
    for s in range(100):
        pi[s] = min(s, 100-s)
    return_track = np.zeros((100, 100))
    return_value = np.zeros((100, 100))

def agent_start(state):
    """
    Hint: Initialize the variables that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global Q, pi, return_track, return_value, episode
    action = rand_in_range(min(state[0], 100 - state[0])) + 1
    return_track[int(state[0])][int(action)] += 1  # increment if the action is taken
    episode = [[int(state[0]),int(action)]]

    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floating point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Q, pi, return_track, return_value, episode
    action = pi[state[0]]
    return_track[int(state[0])][int(action)] += 1  # increment if the action is taken
    return_value[int(state[0])][int(action)] += reward
    episode.append([int(state[0]),int(action)])

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global Q, pi, return_track, return_value, episode

    for pair in episode:
        s = pair[0]
        a = pair[1]
        return_value[s][a] += reward
        Q[s][a] = return_value[s][a] / return_track[s][a]

    for s in range(1, 100):
        optimal = []
        maximum = np.max(Q[s])
        for a in range(1, min(s, 100-s)+1):
            if (Q[s][a] == maximum) and (Q[s][a] != 0):
                optimal.append(a)
        if len(optimal) != 0:
            pi[s] = random.choice(optimal)

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
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"
