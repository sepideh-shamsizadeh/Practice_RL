import numpy as np


def iterative_policy_evaluation(p_state, V, theta, action, rewards, landa):
    delta = 1
    while delta > theta:
        delta = 0
        V1 = V.copy()
        for i in range(0, 4):
            for j in range(0, 4):
                pp = p_state[i][j]
                for s in action:
                    if i == 0 and j == 0:
                        V[i][j] = 0
                    elif i == 3 and j == 3:
                        V[i][j] = 0
                    else:
                        if s == 'up':
                            if i - 1 >= 0:
                                V[i][j] = pp[0] * (rewards + landa * V1[i - 1][j])
                            else:
                                V[i][j] = pp[0] * (rewards + landa * V1[i][j])
                        if s == 'down':
                            if i + 1 <= 3:
                                V[i][j] = pp[1] * (rewards + landa * V1[i + 1][j]) + V[i][j]
                            else:
                                V[i][j] = pp[1] * (rewards + landa * V1[i][j]) + V[i][j]
                        if s == 'left':
                            if j - 1 >= 0:
                                V[i][j] = pp[2] * (rewards + landa * V1[i][j - 1]) + V[i][j]
                            else:
                                V[i][j] = pp[2] * (rewards + landa * V1[i][j]) + V[i][j]
                        if s == 'right':
                            if j + 1 <= 3:
                                V[i][j] = pp[3] * (rewards + landa * V1[i][j + 1]) + V[i][j]
                            else:
                                V[i][j] = pp[3] * (rewards + landa * V1[i][j]) + V[i][j]
                delta = max(delta, np.abs(V[i][j] - V1[i][j]))
        print(V)
    return V


if __name__ == '__main__':
    Vv = np.zeros((4, 4))
    actions = ['up', 'down', 'left', 'right']
    p = [0.25, 0.25, 0.25, 0.25]
    pi = [[p, p, p, p], [p, p, p, p], [p, p, p, p], [p, p, p, p]]
    theta = 0.001
    iterative_policy_evaluation(pi, Vv, theta, actions, -1, 1)
