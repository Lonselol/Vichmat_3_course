#Обратная итерация
import numpy as np
import sympy as sp

A = np.array([
    [72, 9, 8, 5, 1, 1, 2, 5, 6, 1, 7, 1, 3, 9, 9],
    [0, 54, 6, 7, 3, 0, 3, 1, 2, 4, 9, 6, 4, 7, 0],
    [0, 3, 50, 2, 9, 7, 0, 2, 0, 4, 4, 1, 6, 4, 4],
    [8, 2, 0, 70, 4, 4, 3, 9, 6, 0, 7, 5, 6, 7, 6],
    [9, 3, 3, 4, 72, 3, 3, 8, 8, 0, 1, 6, 4, 2, 9],
    [3, 9, 0, 9, 7, 90, 8, 4, 7, 5, 7, 9, 4, 8, 2],
    [3, 5, 5, 9, 8, 7, 74, 5, 5, 9, 0, 5, 3, 0, 2],
    [4, 9, 1, 2, 9, 2, 2, 73, 3, 7, 4, 9, 4, 4, 5],
    [4, 9, 3, 3, 3, 7, 9, 6, 69, 8, 4, 0, 2, 1, 7],
    [2, 5, 4, 1, 8, 7, 5, 3, 5, 61, 4, 3, 6, 0, 1],
    [8, 9, 8, 0, 0, 6, 5, 6, 6, 1, 64, 0, 3, 4, 3],
    [0, 1, 7, 3, 1, 3, 6, 4, 5, 1, 3, 40, 0, 4, 0],
    [0, 4, 6, 7, 2, 7, 5, 1, 1, 1, 6, 6, 64, 8, 0],
    [6, 0, 8, 3, 4, 4, 7, 8, 8, 0, 3, 0, 5, 57, 0],
    [2, 5, 7, 0, 0, 5, 0, 0, 1, 6, 9, 7, 5, 5, 61],
])

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
