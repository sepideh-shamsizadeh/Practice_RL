import numpy as np

actions_pi = {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
rewards = -1
landa = 1
V = np.zeros((4, 4))

teta = 0.001
delta = 1
while delta > teta:
    delta = 0
    V1 = V.copy()
    for i in range(0, 4):
        for j in range(0, 4):
            for s in actions_pi:
                if i == 0 and j == 0:
                    V[i][j] = 0
                elif i == 3 and j == 3:
                    V[i][j] = 0
                else:
                    if s == 'up':
                        if i - 1 >= 0:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i - 1][j])
                        else:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j])
                    if s == 'down':
                        if i + 1 <= 3:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i + 1][j]) + V[i][j]
                        else:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j]) + V[i][j]
                    if s == 'left':
                        if j - 1 >= 0:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j - 1]) + V[i][j]
                        else:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j]) + V[i][j]
                    if s == 'right':
                        if j + 1 <= 3:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j + 1]) + V[i][j]
                        else:
                            V[i][j] = actions_pi.get(s) * (rewards + landa * V1[i][j]) + V[i][j]
            delta = max(delta, np.abs(V[i][j] - V1[i][j]))
    print(V)
