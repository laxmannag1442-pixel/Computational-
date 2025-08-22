# assignment 3
# to code for lu doolittle decomposition
# name:Laxman Nag , Roll no.: 2311092


def lu_decomposition_doolittle(matrix):
  n = len(matrix)
  l=[[0]*n for i in range (n)]
  u=[[0]*n for i in range(n)]

  # Corrected initialization for the first row of u and diagonal of l
  for j in range(n):
      u[0][j] ,l[j][j] = matrix[0][j] ,1
  for j in range(n):
    for i in range(n):      #substituting elements of upper triangular matrix
      sum_u= 0
      if i>=1 and i<=j :
        for k in range(i):
          sum_u+=l[i][k]*u[k][j]
        u[i][j]=matrix[i][j]-sum_u
   
      if i>j:               #substituting elements of lower triangular matrix
        sum_l=0
        for k in range(j):
          sum_l+=l[i][k]*u[k][j]
        l[i][j]=(matrix[i][j]-sum_l)/u[j][j]

  return l , u               #return lower triangular matrix and upper triangular matrix


from lib  import laxmanlibrary
z = laxmanlibrary()
A = z.read_matrix("assignments/matrix_A")
l,u = lu_decomposition_doolittle(A)

print("the given matrix :",A)
print("the lower triangular matrix :",l)
print("upper triangular matrix :",u)
print("the verified matrix after matrix multiplication: ",z.matrix_multiplication(l,u))


#- - -- -- - -- -- - ---- - -- - THE OUPUT RESULTS - --  -- -  - -  - - -- - -- ------------

# the given matrix : [[1, 2, 4], [3, 8, 14], [2, 6, 13]]
# the lower triangular matrix : [[1, 0, 0], [3.0, 1, 0], [2.0, 1.0, 1]]
# upper triangular matrix : [[1, 2, 4], [0, 2.0, 2.0], [0, 0, 3.0]]
# the verified matrix after matrix multiplication:  [[1, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
