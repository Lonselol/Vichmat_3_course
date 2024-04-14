# Вращения с преградами

import numpy as np
import sympy as sp
A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])
n, _ = A.shape
eps = 1e-10
x_col = np.array([sp.Symbol(f"x_{i}") for i in range(n)])
Full = np.hstack((A, x_col[np.newaxis].T))

for i in range(n - 1):
    for j in range(i + 1, n):
        Full[j, :] = Full[j, :] - (Full[j, i] / Full[i, i]) * Full[i, :]

Full_s = [[isinstance(Full[i, j], sp.Expr) for j in range(n + 1)]
    for i in range(n)]

x_next = np.ones(n)
alpha_next = 1
count = 0
diff = 1000000000000
symbols = [[x_col[i], 0] for i in range(n)]

while diff > eps and count < 100000:
    count += 1
    x_pre = x_next
    alpha_pre = alpha_next
    f_pre = x_pre / alpha_pre
    for i in range(n):
        symbols[i][1] = f_pre[i]
    x_next[n - 1] = Full[n - 1, n].subs(symbols) / Full[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x_next[i] = (Full[i, n].subs(symbols) - sum([Full[i, k] * x_next[k] for k in range(i + 1, n)])) / Full[i, i]
    max_x = 0
    for i in range(n):
        if abs(max_x) < abs(x_next[i]):
            max_x = x_next[i]
    alpha_next = max_x
    diff = abs((1 / alpha_pre) - (1 / alpha_next))
lamda = 1 / alpha_next
print("Count: ", count)
validate_lamda = abs(np.linalg.det(A - lamda * np.eye(n)))
print("Diff:", diff)
