import numpy as np


def iterative_policy_evaluation(p_state, V, theta, action, rewards, landa):
    V = np.zeros((4, 4))
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
                                V[i][j] = pp[0] * (rewards[i - 1][j] + landa * V1[i - 1][j])
                            else:
                                V[i][j] = pp[0] * (rewards[i][j] + landa * V1[i][j])
                        if s == 'down':
                            if i + 1 <= 3:
                                V[i][j] = pp[1] * (rewards[i + 1][j] + landa * V1[i + 1][j]) + V[i][j]
                            else:
                                V[i][j] = pp[1] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                        if s == 'left':
                            if j - 1 >= 0:
                                V[i][j] = pp[2] * (rewards[i][j - 1] + landa * V1[i][j - 1]) + V[i][j]
                            else:
                                V[i][j] = pp[2] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                        if s == 'right':
                            if j + 1 <= 3:
                                V[i][j] = pp[3] * (rewards[i][j + 1] + landa * V1[i][j + 1]) + V[i][j]
                            else:
                                V[i][j] = pp[3] * (rewards[i][j] + landa * V1[i][j]) + V[i][j]
                delta = max(delta, np.abs(V[i][j] - V1[i][j]))
    return V


def equalit_marices(a, b):
    eq = True
    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                if a[i][j][k] != b[i][j][k]:
                    return False
    return eq


def policy_iteration(p_state, Vpi, theta, action, rewards, landa):
    policy_notstable = True
    a = 0
    while policy_notstable:
        Vpi = iterative_policy_evaluation(p_state, Vpi, theta, action, rewards, landa)
        old_action = np.copy(p_state)
        for i in range(0, 4):
            for j in range(0, 4):
                mm = []
                for s in action:
                    if i == 0 and j == 0:
                        Vpi[i][j] = 0
                        mm.append(0)
                    else:
                        if s == 'up':
                            if i - 1 >= 0:
                                x = (rewards[i - 1][j] + landa * Vpi[i - 1][j])
                                mm.append(x)
                            else:
                                mm.append(-1000)
                        if s == 'down':
                            if i + 1 <= 3:
                                x = (rewards[i + 1][j] + landa * Vpi[i + 1][j])
                                mm.append(x)
                            else:
                                mm.append(-1000)
                        if s == 'left':
                            if j - 1 >= 0:
                                x = (rewards[i][j - 1] + landa * Vpi[i][j - 1])
                                mm.append(x)
                            else:
                                mm.append(-1000)
                        if s == 'right':
                            if j + 1 <= 3:
                                x = (rewards[i][j + 1] + landa * Vpi[i][j + 1])
                                mm.append(x)
                            else:
                                mm.append(-1000)
                a = np.argmax(mm)
                p = np.zeros(4)
                if i == 0 and j == 0:
                    p[a] = 0
                else:
                    p[a] = 1
                p_state[i][j] = p
        if equalit_marices(old_action, p_state):
            policy_notstable = False

    return p_state, Vpi


if __name__ == '__main__':
    V = np.zeros((4, 4))
    actions = ['up', 'down', 'left', 'right']
    p = [0.25, 0.25, 0.25, 0.25]
    pi = [[p, p, p, p], [p, p, p, p], [p, p, p, p], [p, p, p, p]]
    theta = 0.001
    rewards = -1 * np.ones((4, 4))
    rewards[0][1] = rewards[0][2] = rewards[0][3] = -10
    rewards[2][0] = rewards[2][1] = rewards[2][2] = -10
    ppp, x = policy_iteration(pi, V, theta, actions, rewards, 1)
    pi_states = [['0', '', '', ''], ['', '', '', ''], ['', '', '', ''],['', '', '', '']]
    for k in range(0, 4):
        for l in range(0, 4):
            if l != 0 or k != 0:
                a = np.argmax(ppp[k][l])
                pi_states[k][l] += actions[a]

    print(pi_states)
    print(x)
