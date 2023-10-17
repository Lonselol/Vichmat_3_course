from mylibs import stuff as stuff
from methods import gaussian_elimination as ge

E = 10**(-13)

stuff.generate(3)
#"./txts/matrix.txt"
m = stuff.readMatrix()
methodResult = ge.gaussian_elimination(m[0], m[1])
print(methodResult)
checkResult = stuff.check(m[0], m[1])
print(checkResult)

print(" ")

print(stuff.checkError(methodResult, m[0], m[1], E))
