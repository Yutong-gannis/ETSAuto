import numpy as np


def all_bernstein(n, t):
    b = np.zeros(n+1)
    b[0] = 1.
    t1 = 1.-t
    for j in range(1, n+1):
        saved = 0.
        for i in range(0, j):
            tmp = b[i]
            b[i] = saved + t1*tmp
            saved = t*tmp
        b[j] = saved
    return b


def point_on_bezier_curve(P, t):
    n = len(P) - 1
    b = all_bernstein(n, t)
    c = 0.
    for k in range(0, n+1):
        c += b[k]*P[k]
    return c
