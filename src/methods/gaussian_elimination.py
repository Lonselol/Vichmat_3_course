import numpy as np

def gaussian_elimination(A, b, n):
    iter = 0
    while iter < n:
        maxx = A[iter, iter]
        i = iter
        j = 0
        
        # абсолютный максимум
        for l in range(iter, n):
            for k in range(n):
                if abs(maxx) < abs(A[l, k]):
                    maxx = A[l, k]
                    i = l
                    j = k
        
        # Поменяться местами
        A[[iter, i]] = A[[i, iter]]
        b[[iter, i]] = b[[i, iter]]
        
        # Работайте братья
        for l in range(iter + 1, n):
            if A[iter, j] != 0:  # НЫА ноль
                m = A[l, j] / A[iter, j]
                A[l] -= m * A[iter]
                b[l] -= m * b[iter]
        
        if maxx != 0:  # На ноль
            A[iter] /= maxx
            b[iter] /= maxx
        
        iter += 1
    
    # Обратный ход
    ans = np.zeros(n)
    
    for k in range(1, n + 1):
        for l in range(n):
            if A[n - k, l] == 1:
                ans[l] = b[n - k]
                i = l
        
        for k in range(n):
            b[k] -= ans[i] * A[k, i]
            A[k, i] = 0
    
    return ans