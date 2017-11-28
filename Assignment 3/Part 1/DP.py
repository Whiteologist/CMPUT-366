import numpy as np
import matplotlib.pyplot as plt

p_h = 0.40
theta = 2e-256

Value = np.zeros(101)
Total = np.zeros((101,4))
Policy = np.zeros(101)
Reward = np.zeros(101)
Reward[100] = 1.0
sweep = 0

while True:
    delta = 0.0
    for state in range(1, 100):
        value = Value[state]
        Returns = []
        for action in range(1, min(state, 100 - state) + 1):
            Returns.append(p_h * (Reward[state + action] + Value[state + action]) +\
                    (1 - p_h) * (Reward[state - action] + Value[state - action]))
        Value[state] = np.max(Returns)
        Policy[state] = np.argmax(Returns) + 1
        delta = max(delta, abs(value - Value[state]))
        if sweep < 3:
            Total[state][sweep] = Value[state]
        else:
            Total[state][3] = Value[state]
    if delta < theta:
        break
    sweep += 1
for i in range(0, 4):
    Total[0][i] = None
    Total[100][i] = None
Policy[0] = None
Policy[100] = None

plt.figure(1)
plt.suptitle('Gambler problem (p = ' + str(p_h) + ')')
plt.xlabel('Capital')
plt.ylabel('Value\nestimates')
plt.plot(Total)
plt.figure(2)
plt.suptitle('Gambler problem (p = ' + str(p_h) + ')')
plt.xlabel('Capital')
plt.ylabel('Final\npolicy\n(stake)')
plt.plot(Policy)
plt.show()
