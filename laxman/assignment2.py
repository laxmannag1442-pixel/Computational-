# Name =laxman nag ,roll number = 2311092
from lib import laxmanlibrary
import matplotlib.pyplot as plt

z = laxmanlibrary()
N =[20,1000,10000,15000,20000]
pi_l =[]
for i in N:
    list1 = z.lcg(0.8,i)
    list2 = z.lcg(0.1, i)
    inside_qc_x = []
    inside_qc_y = []
    inside_sq_x = []
    inside_sq_y = []
    n_inside_circle = 0
    n_inside_square = 0

    for x,y in zip(list1,list2):
        if x * y <= 1:
            inside_sq_x.append(x)
            inside_sq_y.append(y)
            if x**2 + y**2 <= 1:
                inside_qc_x.append(x)
                inside_qc_y.append(y)
                n_inside_circle+=1
        else:
            pass

    pi = (4*n_inside_circle)/i
    pi_l.append(pi)

print(pi_l)
plt.hist(pi_l,bins=40)
plt.show()
