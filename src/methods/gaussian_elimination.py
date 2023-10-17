import numpy as np

#Схема Гаусса с выбором главного элемента

def gaussian_elimination(A, b):
    n = len(A)
    
    for i in range(n):
        # Выбор главного элемента в оставшейся матрице по модулю
        max_idx = np.abs(A[i:, i]).argmax() + i
        A[[i, max_idx]] = A[[max_idx, i]]
        b[[i, max_idx]] = b[[max_idx, i]]
        
        # Треугольник
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i] #текущий на главный и нули ниже главный (1-1=0)
            A[j] -= ratio * A[i]
            b[j] -= ratio * b[i]
        print (A[i][i])

    # Обратный ход - находим решение системы, дот - скалярное для суммы
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]

    return x
