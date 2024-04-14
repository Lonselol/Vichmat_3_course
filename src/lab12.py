#Простая итерация
import numpy as np

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])
n, _ = A.shape
eps = 1e-10
x_next = np.ones(n)
count = 0
diff = eps + 100

while diff > eps:
    count += 1
    x_prev = x_next
    y = np.matmul(A, x_prev)
    lamda = np.dot(y, x_prev)
    x_next = y / np.linalg.norm(y)
    sign = 1 if lamda > 0 else -1
    diff = abs(sign * x_next - x_prev).max()

print("Count: ", count)
print("Diff:", diff)
