from mylibs import stuff as stuff
from methods import gaussian_elimination as ge

E = 10**(-13)

#Делаем матрицу
stuff.generate(7)
m = stuff.readMatrix("./txts/generated.txt")
n = len(m[0])
#Проверяем результаты
checkResult = stuff.check(m[0], m[1])
print("Результат\n", checkResult)

#Методы
methodResult = ge.gaussian_elimination(m[0], m[1], n)
print("Результат метода\n", methodResult)

#Ошибки
check = (stuff.checkError(methodResult, m[0], m[1], E))
print ("Ошибки")
for index in check:
  print (check.get(index))
