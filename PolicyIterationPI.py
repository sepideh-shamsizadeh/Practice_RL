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
                    else:
                        if s == 'up':
                            if i - 1 >= 0:
                                V[i][j] = pp[0] * (rewards[i][j] + landa * V1[i - 1][j])
                            else:
                                V[i][j] = pp[0] * (rewards[i][j] + landa * V1[i][j])
                        if s == 'down':
                            if i + 1 <= 3:
                                V[i][j] = pp[1] * (rewards[i][j] + landa * V1[i + 1][j]) + V[i][j]
                            else:
                                V[i][j] = pp[1] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                        if s == 'left':
                            if j - 1 >= 0:
                                V[i][j] = pp[2] * (rewards[i][j] + landa * V1[i][j - 1]) + V[i][j]
                            else:
                                V[i][j] = pp[2] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                        if s == 'right':
                            if j + 1 <= 3:
                                V[i][j] = pp[3] * (rewards[i][j] + landa * V1[i][j + 1]) + V[i][j]
                            else:
                                V[i][j] = pp[3] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                delta = max(delta, np.abs(V[i][j] - V1[i][j]))
    return V


def equalit_marices(a, b):
    eq = True
    for i in a:
        for j in b:
            if i != j:
                return False
    return eq


def policy_iteration(p_state, Vpi, theta, action, rewards, landa):
    policy_notstable = True
    a = 0
    while policy_notstable:
        Vpi = iterative_policy_evaluation(p_state, Vpi, theta, action, rewards, landa)
        for i in range(0, 4):
            for j in range(0, 4):
                old_action = p_state[i][j]
                for s in action:
                    if i == 0 and j == 0:
                        Vpi[i][j] = 0
                    else:
                        if s == 'up':
                            if i - 1 >= 0:
                                max = Vpi[i - 1][j]
                                a = 0
                            else:
                                max = Vpi[i][j]
                                a = 0
                        if s == 'down':
                            if i + 1 <= 3:
                                if Vpi[i + 1][j] > max:
                                    max = Vpi[i + 1][j]
                                    a = 1
                            else:
                                if Vpi[i][j] > max:
                                    max = Vpi[i][j]
                                    a = 1
                        if s == 'left':
                            if j - 1 >= 0:
                                if Vpi[i][j - 1] > max:
                                    max = Vpi[i][j - 1]
                                    a = 2
                            else:
                                if Vpi[i][j] > max:
                                    max = Vpi[i][j]
                                    a = 2
                        if s == 'right':
                            if j + 1 <= 3:
                                if Vpi[i][j + 1] > max:
                                    max = Vpi[i][j + 1]
                                    a = 3
                            else:
                                if Vpi[i][j] > max:
                                    max = Vpi[i][j]
                                    a = 3

                p = np.zeros(4)
                if i == 0 and j == 0:
                    p[a] = 0
                else:
                    p[a] = 1
                p_state[i][j] = p
                if equalit_marices(old_action, p_state[i][j]):
                    policy_notstable = False
    return p_state, Vpi


if __name__ == '__main__':
    Vv = np.zeros((4, 4))
    actions = ['up', 'down', 'left', 'right']
    p = [0.25, 0.25, 0.25, 0.25]
    pi = [[p, p, p, p], [p, p, p, p], [p, p, p, p], [p, p, p, p]]
    theta = 0.001
    rewards = -1 * np.ones((4, 4))
    rewards[0][1] = rewards[0][2] = rewards[0][3] = -10
    rewards[2][0] = rewards[2][1] = rewards[2][2] = -10
    policy_iteration(pi, Vv, theta, actions, rewards, 1)
