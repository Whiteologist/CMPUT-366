#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
"""

import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_axis = []
    y_axis = []

    data = open("output2.txt", "r")
    for line in data:
        line = line.strip()
        x = float(line.split(",")[0])
        y = int(line.split(",")[1])
        x_axis.append(x)
        y_axis.append(y)

    plt.plot(x_axis, y_axis, label="epsilon = 0.06")
    plt.xlabel('Step sizes')
    # plt.xticks([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0])
    plt.ylabel('Average # of steps per episode')
    plt.legend()
    plt.show()
