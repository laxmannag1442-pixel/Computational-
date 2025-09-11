# Name-Laxman Nag ,Roll number-2311092
# My Own Library for future use

import numpy as np
import matplotlib.pyplot as plt

class laxmanlibrary:

  # to read matrix from file 
  def read_matrix(self,filename):
      with open(filename,'r') as file:
        matrix = []
        for line in file:
        # Split the line into numbers and convert to int/float
          row =[float(num)for num in line.strip().split()]
          matrix.append(row)
        return matrix

  # to create a list of uniformly_pRNG between [0,1)      
  def lcg(self,seed,n):
    a = 1103515245 
    c = 12345
    m = 32768
    list =[]
    for _ in range(int(n)):
      seed = (a*seed +c)%m
      list.append(seed/m)
    return list
  
  # to create a list of uniformly_pRNG between [min_value,max_value)
  def uniformly_pRNG(self,seed,n,min_value,max_value):
    list = self.lcg(seed,n)
    new_l =[]
    for i in list:
      i = min_value+(max_value-min_value)*i
      new_l.append(i)
    return new_l
  

  # to create a list of exponentially pRNG  
  def exponentially_pRNG(self,seed,n,a):
    list = self.lcg(seed,n)
    new_l =[]
    for i in list:
      i = (-1)*np.log(i)/a
      new_l.append(i)
    return new_l


  # to plot the correlation of pRNG
  def correlation_plot(self,list,k):
    x =list[k:]
    y =list[:-k]
    plt.scatter(x,y,marker ="o",color ="blue")
    plt.title(rf"correlation ploting between $x_i$ and $x_{{i+{k}}}$")
    plt.xlabel("$x_i$")
    plt.ylabel(rf"$x_{{i+{k}}}$")
    plt.grid(True)
    plt.show()

  #to solve linear equation using gauss jordan 

  def make_augmented_matrix(self, A, B):
      matrix = []
      for row, b_row in zip(A, B):      # clearer name
          matrix.append(row + b_row)    # since b_row is [value], concat works
      return matrix

  def rows_swap(self, matrix, i, j):
      if i != j:
          matrix[i], matrix[j] = matrix[j], matrix[i]
      return matrix

  def scaled_and_add_rows(self, matrix, oper_row, ref_row, scalar):
      for i in range(len(matrix[oper_row])):
          matrix[oper_row][i] -= matrix[ref_row][i] * scalar
      return matrix

  def find_row_num(self, matrix, column):
      # Partial pivoting
      max_val = 0
      row_number = column
      for i in range(column, len(matrix)):
          if abs(matrix[i][column]) > abs(max_val):
              max_val = matrix[i][column]
              row_number = i
      return row_number, max_val

  def scaled(self, matrix, row_num, column_num):
      v = matrix[row_num][column_num]
      if v != 0:
          for i in range(len(matrix[0])):
              matrix[row_num][i] /= v
      return matrix

  def gauss_jordan(self, A, b, tol=1e-12):
    matrix = self.make_augmented_matrix(A, b)
    n = len(A)

    for j in range(n):
        row, pivot = self.find_row_num(matrix, j)
        if abs(pivot) < tol:   # check tiny pivots
            raise ValueError("Jacobian is singular or nearly singular at this step.")
        if row != j:
            self.rows_swap(matrix, j, row)

        self.scaled(matrix, j, j)

        for z in range(n):
            if z != j:
                self.scaled_and_add_rows(matrix, z, j, matrix[z][j])

    return [row[-1] for row in matrix]

  # code for matrix multiplication
  def matrix_multiplication(self,a,b):
        m =len(a)
        n = len(b[0])
        c =[[1]*n for _ in range(m)]
        for i in range(m):
          for j in range(n):
            sum =0
            for k in range(len(b)):
              sum+=a[i][k]*b[k][j]
            c[i][j] =  sum 
      
        return c # return  the matrix multiplication of a and b

  
  #to code for lu decomposition by doolittle
  def lu_decomposition_doolittle(self,matrix):
    n = len(matrix)
    l=[[0]*n for i in range (n)]
    u=[[0]*n for i in range(n)]
  
    # Corrected initialization for the first row of u and diagonal of l
    for j in range(n):
        u[0][j] ,l[j][j] = matrix[0][j] ,1
    for j in range(n):
      for i in range(n):       
        sum_u= 0
        if i>=1 and i<=j :      #substituting elements of upper triangular matrix
          for k in range(i):
            sum_u+=l[i][k]*u[k][j]
          u[i][j]=matrix[i][j]-sum_u 
      
        if i>j:                  #substituting elements of lower triangular matrix
          sum_l=0
          for k in range(j):
            sum_l+=l[i][k]*u[k][j]
          l[i][j]=(matrix[i][j]-sum_l)/u[j][j]
  
    return l , u    #return lower triangular matrix and upper triangular matrix

  def cholesky_decompose(self,A):
      n = len(A)
      L = [[0.0 for _ in range(n)] for _ in range(n)]
      for i in range(n):
          for j in range(i + 1):
              s = A[i][j]
              for k in range(j):
                  s -= L[i][k] * L[j][k]
              if i == j:
                  if s <= 0:
                      raise ValueError("Matrix not positive definite")
                  L[i][j] = (s ** 0.5)
              else:
                  L[i][j] = s / L[j][j]
      return L

  def cholesky_solve(self,L, b):
      n = len(L)
      y = [[0.0] for _ in range(n)]
      for i in range(n):
          s = b[i][0]
          for j in range(i):
              s -= L[i][j] * y[j][0]
          y[i][0] = s / L[i][i]
      x = [[0.0] for _ in range(n)]
      for i in range(n - 1, -1, -1):
          s = y[i][0]
          for j in range(i + 1, n):
              s -= L[j][i] * x[j][0]
          x[i][0] = s / L[i][i]
      return x

  def jacobi(self,A, b, tol=1e-6, max_iter=10000):
      n = len(A)
      x_old = [[0.0] for _ in range(n)]
      x_new = [[0.0] for _ in range(n)]
      
      for it in range(max_iter):
          for i in range(n):
              s = sum(A[i][j]*x_old[j][0] for j in range(n) if j != i)
              x_new[i][0] = (b[i][0] - s) / A[i][i]
          
          # check convergence (infinity norm of difference)
          if max(abs(x_new[i][0]-x_old[i][0]) for i in range(n)) < tol:
              return x_new, it+1
          
          x_old = x_new[:]
      
      raise ValueError("Jacobi did not converge within max_iter")
  
  def gauss_seidel(self,A, b, tol=1e-6, max_iter=10000):
    n = len(A)
    x = [[0.0] for _ in range(n)]
    for it in range(max_iter):
        l=[]        
        for i in range(n):
            s1 = sum(A[i][j]*x[j][0] for j in range(i))
            s2 = sum(A[i][j]*x[j][0] for j in range(i+1, n))
            v = (b[i][0] - s1 - s2) / A[i][i]
            l.append(abs(v - x[i][0]))
            x[i][0] = v
        if max(l) < tol:
            return x, it + 1
    raise ValueError("Gauss-Seidel method did not converge")
  
  # to find root of function by bisection method
  def bisection_method(self, f, a, b, tol=1e-6, max_iter=1000):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or abs(b - a) < tol:
            return c, i+1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return None, i+1


  def regula_falsi_method(self, f, a, b, tol=1e-6, max_iter=1000):
    if f(a) * f(b) > 0:
        print("Regula Falsi method fails.")
        return None
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        c = b - (fb * (a - b)) / (fa - fb)
        if abs(f(c)) < tol:
            return c, i+1
        if fa * f(c) < 0:
            b = c
        else:
            a = c
    return None, i+1


  def bracketing(self, g, a, b, beta=0.5, max_iter=100):
    i = -1
    for i in range(max_iter):
        if g(a) * g(b) < 0:
            return a, b
        if abs(g(a)) < abs(g(b)):
            a = a - beta * (b - a)
        else:
            b = b + beta * (b - a)
    return None, i+1


  def newton_raphson(self, f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative zero. Newton-Raphson fails.")
        c = x - f(x) / dfx
        if abs(c - x) < tol:
            return c, i+1
        x = c
    return None, i+1


  def fix_point(self, f, x0, tol=1e-6, max_iter=10000):
    x = x0
    for i in range(max_iter):
        c = f(x)
        if abs(c - x) < tol:
            return c, i+1
        x = c
    return None, i+1


  def fixed_point_polynomial(self, x, g, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x_new = [f(*x) for f in g]
        sum1 = sum(val**2 for val in x)
        sum2 = sum(val**2 for val in x_new)
        if sum2 == 0 or abs((sum2 - sum1) / sum2) < tol:
            return x_new, i+1
        x = x_new
    return None, i+1


  def newton_raphson_multivariable(self, x0, f, jacobian, tol=1e-6, max_iter=100, alpha=1.0):
    for i in range(max_iter):
        fun = [[-fi(*x0)] for fi in f]
        J = jacobian(x0)
        delta = self.gauss_jordan(J, fun)
        x_new = [x0[j] + alpha * delta[j] for j in range(len(x0))]
        # compute residual norm
        res = np.linalg.norm([fi(*x_new) for fi in f])
        if res < tol:
            return x_new, i+1
        x0 = x_new
    return None, i+1

      
    
    
