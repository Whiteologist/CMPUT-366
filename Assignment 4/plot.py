#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
"""

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = open("output.txt", "r")
    x_axis = []
    y_axis = []

    for line in data:
        line = line.strip()
        x_axis.append(int(line.split(",")[0]))
        y_axis.append(int(line.split(",")[1]))

    plt.plot(x_axis, y_axis)
    plt.xlim([0, 8000])
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.show()
