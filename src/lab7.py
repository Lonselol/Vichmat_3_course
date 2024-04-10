import math
import numpy as np

np.set_printoptions(precision=7)

A = np.array([[2.12, 0.48, 1.34, 0.88, 11.172],
              [0.42, 3.95, 1.87, 0.43, 0.115],
              [1.34, 1.87, 2.98, 0.46, 9.009],
              [0.88, 0.43, 0.46, 4.44, 9.349]]).astype(float)
AllMatrix = [A]
print(A)
RMatrix = []
n = len(A)


def vectornormali(a, k):
    pStepkB = np.zeros(k)  # [0]
    pStepkK = np.zeros(n - k)  # [].. size = 2 [0,0]

    a = [row[k] for row in a]
    a = a[k:n]

    for i in range(n - k):  # 0..2  #1...2
        if (i == 0):
            if (a[i] >= 0):
                sigmaK = 1
                pStepkK[0] = a[i] + sigmaK * (math.sqrt(sum(a[l] ** 2 for l in range(i, n - k))))
            else:
                sigmaK = -1
                pStepkK[0] = a[i] + sigmaK * (math.sqrt(sum(a[l] ** 2 for l in range(i, n - k))))
        else:
            pStepkK[i] = a[i]  # pStepkK[1] = a[1][0]

    pStepk = np.hstack((pStepkB, pStepkK))
    return pStepk


def solveR(aB, k, r, pStepk):
    for i in range(k + 1, n + 1):  # 1,2,3
        aForDot = [row[i] for row in aB]
        for j in range(n):  # 0,1,2
            r[i - 1][j] = 2 * pStepk.dot(aForDot) * pStepk[j] / (sum(pStepk[l] ** 2 for l in range(k, n)))
    r = np.transpose(r)

    return r


def solveA(aB, pStepk, k):
    newA = np.zeros((n, n + 1))
    if (k > 0):
        for i in range(0, k):
            newA[i] = aB[i]

    sigma = 1
    newA[k][k] = -sigma * math.sqrt(sum(aB[l][k] ** 2 for l in range(k, n)))

    for i in range(k, n):
        for j in range(k + 1, n + 1):
            newA[i][j] = aB[i][j] - 2 * pStepk[i] * (
                    (sum(pStepk[l] * aB[l][j] for l in range(k, n))) / (sum(pStepk[l] ** 2 for l in range(k, n))))
    return newA


def solveX(g, r):
    x = np.zeros(n)

    x[n - 1] = g[n - 1] / r[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (g[i] - sum(r[i][j] * x[j] for j in range(i + 1, n))) / r[i][i]

    return x


for k in range(n - 1):
    aB = np.copy(A)
    pStepk = vectornormali(aB, k)
    s = np.linalg.norm((pStepk)) ** 2

    r = np.zeros((n, n))
    r = solveR(aB, k, r, pStepk)
    RMatrix.append(r)

    A = solveA(aB, pStepk, k)
    AllMatrix.append(A)

A = AllMatrix[n - 1]
R = A[:, :n]
b = [row[-1] for row in A]
x = solveX(b, R)
print("Решение: ", x)

A = AllMatrix[0]
a = A[:, :n]
b = [row[-1] for row in A]
print(np.linalg.solve(a, b))

solution = np.dot(a, x)

print("Погрешность: ", max(abs(solution - b)))