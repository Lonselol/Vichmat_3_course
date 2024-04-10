# Ричардсон
import math

import numpy as np

class Rotation:
    def __init__(self, n: int, mtx: np.ndarray, p: int):
        self.k = 0
        self.n = n
        self.mtx = mtx
        self.p = p

    def sign(self, num):
        if num > 0:
            return 1
        return -1

    def index_of_largest(self, matrix):
        largest = abs(matrix[0][1])
        i_l, j_l = 0, 1
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if abs(matrix[i, j]) > largest:
                    largest = abs(matrix[i, j])
                    i_l, j_l = i, j
        return i_l, j_l

    def check_tol(self, matrix):

        # Преграда
        self.tol = np.sqrt(max(abs(np.diag(matrix)))) * (10 ** (-self.k))

        #Проверка внедиагональных элементов
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if abs(matrix[i, j]) >= self.tol:
                    return False

        if self.k < self.p:
            self.k += 1
            return False

        return True

    def solve(self):
        mtx = self.mtx
        steps = 0

        while not self.check_tol(mtx):
            q, p = self.index_of_largest(mtx)

            d = abs(mtx[p, p] - mtx[q, q]) / np.sqrt(
                (mtx[p, p] - mtx[q, q]) ** 2 + 4 * mtx[p, q] ** 2
            )

            c = np.sqrt(0.5 * (1 + d))
            s = self.sign(mtx[p, q] * (mtx[p, p] - mtx[q, q])) * np.sqrt(0.5 * (1 - d))

            R = np.eye(self.n)
            R[p, p] = R[q, q] = c
            R[p, q] = -s
            R[q, p] = s
            mtx = R.T.dot(mtx).dot(R)
            mtx[p, q] = mtx[q, p] = 0
            steps += 1

        return steps, np.diag(mtx)

def getMaxError(mat, x, vector):
    errors = mat.dot(x) - vector
    return np.max(np.abs(errors))


n = 2
mtx = np.array([[2,1],[1,2]])
vector = np.array([4, 5])

e = 1e-11

n = mtx.shape[0]
ans = np.zeros(n)
eigs = Rotation(n, mtx, 20).solve()[1]

lmin = np.min(eigs)
lmax = np.max(eigs)

tau0 = 2 / (lmin + lmax)
nnn = lmin / lmax

p0 = (1 - nnn) / (1 + nnn)
p1 = (1 - math.sqrt(nnn)) / (1 + math.sqrt(nnn))

maxiters = np.log(2 / e) / np.log(1 / p1)
print(maxiters)

steps = 0
tau = tau0
iters = round(maxiters) + 1

while steps < iters:
    steps += 1
    v = np.cos(2 * steps - 1) * np.pi / (2 * maxiters)
    tau = tau0 / (1 + p0 * v)
    ans -= tau * (mtx.dot(ans) - vector)

    if steps == iters:
        if getMaxError(mtx, ans, vector) > e:
            iters += (round(maxiters)+1)


print("Steps:\n", steps, "\n\n")
print("Answer(x):\n", ans, "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")
