import numpy as np

#Рид филе
def readMatrix(fileName = "./txts/generated.txt"):
  file_path = fileName
  data = np.loadtxt(file_path)
  A = data[:, :-1]
  b = data[:, -1]
  return A, b

#Дегенерировать СЛАУ
def generate(size):
  A = np.random.rand(size, size)
  b = np.random.rand(size)
  data = np.column_stack((A, b))
  np.savetxt('./txts/generated.txt', data)

#Проверка
def check(A, b):
  return np.linalg.solve(A,b)

#Проверка погрешности
def checkError(result, A, b, error):
  result = (np.dot(A, result) - b) <= error
  return result
