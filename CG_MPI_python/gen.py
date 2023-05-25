import numpy as np
import sys 

args = sys.argv

n = int(args[1])
A = np.random.rand(n, n)
A = (A + A.T) / 2
for i in range(n):
    A[i, i] = A[i, :].sum()
x_sol = np.random.rand(n)
b = A @ x_sol

np.savetxt("A.txt", A)
np.savetxt("b.txt", b)


