,# Name-Laxman Nag ,Roll number-2311092
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
  def make_augmented_matrix(self,A,B):
    matrix =[]
    for row,b in zip(A,B):
        matrix.append(row + b)
    return matrix

  def rows_swap(self,matrix , i , j):
    matrix[i], matrix[j] = matrix[j] , matrix[i]
    return matrix

  def scaled_and_add_rows(self,matrix,oper_row,ref_row,scalar):
    for i in range(len(matrix[oper_row])):
        matrix[oper_row][i] = matrix[oper_row][i] - matrix[ref_row][i]* scalar
    return matrix

  def find_row_num(self,matrix,column):
    max = 0
    for row in matrix:
        if row[column]>max:
            max =row[column]
    pivot_element = max

    for i in range(len(matrix)):
        if matrix[i][column] ==max:
            row_number = i
            return row_number,pivot_element
        
  def scaled(self,matrix,row_num,column_num):
    v = matrix[row_num][column_num]
    for i in range(len(matrix[0])):
        matrix[row_num][i] = matrix[row_num][i]/v
    return matrix
    
  def find_solution(self,A,b):
    matrix = self.make_augmented_matrix(A,b)
    for j in range(len(matrix[0])):    
        row , pivot = self.find_row_num(matrix,j)
        i = j
        if i <=len(matrix)-1 :
            if i!=row: self.rows_swap(matrix,i,row) 
            self.scaled(matrix,i,j)
            for z in range(len(matrix)):
                if z!=i:
                    self.scaled_and_add_rows(matrix,z,i,matrix[z][j])
    sol = []
    for row in matrix:
        sol.append(row[-1])
    return sol  # return the solution set

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
      
    
    
