from mylibs import stuff as stuff
from methods import gaussian_elimination as ge
from methods import reflections_first as rf
import math

E = 10**(-13)
I = 1
NGEN = 10
MATRIX = "./txts/generated.txt"

#Делаем матрицу

errors = []

for i in range(I):
  print ("Итерация: ", i)
  stuff.generate(NGEN)
  m = stuff.readMatrix(MATRIX)
  #Проверяем результаты
  checkResult = stuff.check(m[0], m[1])
  print("Результат\n", checkResult)

  #Методы
  #methodResult = ge.gaussian_elimination(m[0], m[1])
  #print("Метод Гаусса с выбором главного\n", methodResult)
  #Отражений_1
  methodResult = rf.reflections_first(m[0], m[1])
  print("Метод отражений\n", methodResult)

  #Ошибки
  check = (stuff.checkError(methodResult, m[0], m[1], E))
  for index in check:
    a = check.get(index)[0]
    if a != 0:
      a = round((math.log10(a)), 1)
      errors.append(a)

stuff.createBarChart(errors)
