from mylibs import stuff as stuff
from methods import gaussian_elimination as ge

E = 10**(-13)
n = 10

#stuff.generate(10)
m = stuff.readMatrix("./txts/matrix.txt")
methodResult = ge.gaussian_elimination(m[0], m[1])
print(methodResult)
checkResult = stuff.check(m[0], m[1])
print(checkResult)

print(" ")

print(stuff.checkError(methodResult, m[0], m[1], E))
