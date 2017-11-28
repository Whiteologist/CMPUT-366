#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
"""

import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_axis = []
    y_axis = []

    data = open("output1.txt", "r")
    for line in data:
        line = line.strip()
        x = int(line.split(",")[0])
        y = int(line.split(",")[1])
        x_axis.append(x)
        y_axis.append(y)

    plt.plot(x_axis[1:50], y_axis[1:50], c="b", label="n = 0")
    plt.plot(x_axis[51:100], y_axis[51:100], c="g", label="n = 5")
    plt.plot(x_axis[101:150], y_axis[101:150], c="r", label="n = 50")
    plt.xlim([0, 50])
    plt.ylim([0, 800])
    plt.xticks([2, 10, 20, 30, 40, 50])
    plt.yticks([14, 200, 400, 600, 800])
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()
    plt.show()
