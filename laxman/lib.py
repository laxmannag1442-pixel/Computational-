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
    matrix = make_augmented_matrix(A,b)
    for j in range(len(matrix[0])):    
        row , pivot = find_row_num(matrix,j)
        i = j
        if i <=len(matrix)-1 :
            if i!=row: rows_swap(matrix,i,row) 
            scaled(matrix,i,j)
            for z in range(len(matrix)):
                if z!=i:
                    scaled_and_add_rows(matrix,z,i,matrix[z][j])
    sol = []
    for row in matrix:
        sol.append(row[-1])
    return sol

      
    
    
