import numpy as np

def thomas(alpha, beta, gamma, f):
    n = len(f)
    v = np.zeros(n)
    [aa, dd, cc, bb] = map(np.array, [alpha, beta, gamma, f])
    for i in range(1, n):
        mult = aa[i]/dd[i-1]
        dd[i] = dd[i] - mult*cc[i-1]
        bb[i] = bb[i] - mult*bb[i-1]
    v[n-1] = bb[n-1]/dd[n-1]
    for i in range(n-2, -1, -1):
        v[i] = (bb[i] - cc[i]*v[i+1])/dd[i]
    return v
