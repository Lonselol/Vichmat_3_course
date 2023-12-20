import numpy as np
import math

def vectornormali(a,k,n):

    p_stepk_ = np.zeros(k) #[0]
    p_stepk_k_ = np.zeros(n-k) #[].. size = 2 [0,0]

    a = [row[k] for row in a]
    a = a[k:n]

    for i in range(n-k): #0..2  #1...2
        if(i==0):
            if(a[i]>= 0):
                sigma_k = 1
                p_stepk_k_[0] = a[i] + sigma_k * (math.sqrt(sum(a[l] ** 2 for l in range(i, n-k))))
            else:
                sigma_k = -1
                p_stepk_k_[0] = a[i] + sigma_k * (math.sqrt(sum(a[l] ** 2 for l in range(i, n-k))))
        else:
            p_stepk_k_[i] = a[i] #p_stepk_k_[1] = a[1][0]


    p_stepk = np.hstack((p_stepk_,p_stepk_k_))
    return p_stepk

def solve_r(a_,k,r,p_stepk,n):

    for i in range(k+1,n+1):#1,2,3
        a_for_dot = [row[i] for row in a_]
        for j in range(n): #0,1,2
                r[i-1][j] = 2 * p_stepk.dot(a_for_dot)*p_stepk[j] / (sum(p_stepk[l] ** 2 for l in range(k, n)))
    r = np.transpose(r)

    return r

def solve_a(a_,p_stepk,k,n):

    new_a = np.zeros((n,n+1))
    if (k > 0):
        for i in range(0,k):
            new_a[i] = a_[i]

    sigma = 1
    new_a[k][k] = -sigma * math.sqrt(sum(a_[l][k]**2 for l in range(k,n)))

    for i in range(k,n):
        for j in range(k+1,n+1):
             new_a[i][j] = a_[i][j] - 2 * p_stepk[i] * ((sum(p_stepk[l]*a_[l][j] for l in range(k,n)))/(sum(p_stepk[l]**2 for l in range(k,n))))
    return new_a

def solve_x(g,r,n):
    x = np.zeros(n)

    x[n-1] = g[n-1]/r[n-1][n-1]

    for i in range(n-2,-1,-1):
        x [i] = (g[i] - sum(r[i][j]*x[j] for j in range (i+1,n)))/ r[i][i]


    return x

def reflections_first(matrix: np.matrix, values: np.array):
  A = np.column_stack((matrix, values))
  n = len(matrix)
  AllMatrix = [A]
  RMatrix = []

  for k in range(n-1):

      a_ = np.copy(A) #Копируем А
      p_stepk = vectornormali(a_,k,n) #Вектор нормалей

      r = np.zeros((n,n))
      r = solve_r(a_,k,r,p_stepk,n)
      RMatrix.append(r)

      A = solve_a(a_,p_stepk,k,n)
      AllMatrix.append(A)

  A = AllMatrix[n-1] #Срез для итоговой
  R = A[:,:n]
  b = [row[-1] for row in A]
  solution = solve_x(b,R,n)
  return solution
