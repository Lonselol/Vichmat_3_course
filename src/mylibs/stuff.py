import numpy as np
import matplotlib.pyplot as plt

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
  res = {}
  errors = np.abs(np.dot(A, result) - b) # <= error
  for index, item in enumerate(errors):
    res.update({index: [item, item<=error]})
  return res

#Гистограмма решений
def createBarChart(errors):
  # This is a list of unique values appearing in the input list
  errors_unique = list(set(errors))
  # This is the corresponding count for each value
  counts = [errors.count(value) for value in errors_unique]
  # Some labels and formatting to look more like the example
  #plt.bar_label(barcontainer, errors_unique, label_type='edge')
  plt.hist(errors, bins=len(errors_unique)*2, edgecolor='black')
  plt.ylabel("Количество элементов решений с точностью")
  plt.xlabel("Точность")
  plt.title("")
  plt.show()