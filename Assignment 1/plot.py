import matplotlib.pyplot as plt
import csv

x = []

with open('RL_EXP_OUT.dat','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        x.append(row[0])

plt.ylim(ymax = 1.0)
plt.plot(x, label='alpha = 0.1\nepsilon = 0.0\nQ_1 = 5')
plt.legend()
plt.show()
