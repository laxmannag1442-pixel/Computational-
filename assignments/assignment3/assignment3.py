from lib  import laxmanlibrary
z = laxmanlibrary()
A = z.read_matrix("matrix_A")
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
