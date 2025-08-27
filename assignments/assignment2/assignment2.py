from lib import laxmanlibrary

z = laxmanlibrary()
A =z.read_matrix("matrix_m")
B =z.read_matrix("matrix_n")
print(f"matrix A :{A}")
print(f"matrix B :{B}")

solution =z.find_solution(A,B)
print(f"solution : {solution}")
