import numpy as np

#Схема Гаусса с выбором главного элемента

def gaussian_elimination(A, b):
    n = len(A)
    
    for i in range(n):
        # Выбор главного элемента в итерации в толбце по модулю
        max_idx = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_idx][i]):
                max_idx = j
        A[[i, max_idx]] = A[[max_idx, i]]
        b[[i, max_idx]] = b[[max_idx, i]]
        
        # Приведение матрицы к треугольному виду
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i] #текущий на главный и нули ниже главный
            A[j] -= ratio * A[i]
            b[j] -= ratio * b[i]

    # Обратный ход - находим решение системы, дот - скалярное для суммы
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]

    return x