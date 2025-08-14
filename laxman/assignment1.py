#code to find pi values and generating exponential distribution fron pRNG
# Name =laxman nag ,roll number = 2311092


from lib  import laxmanlibrary
import matplotlib.pyplot as plt

z = laxmanlibrary()
N =[500,5000,1000,10000,15000,20000]
pi_l =[]
for i in N:
    list1 = z.lcg(89, i)
    list2 = z.lcg(56, i)
    inside_qc_x = []
    inside_qc_y = []
    inside_sq_x = []
    inside_sq_y = []
    n_inside_circle = 0
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

for i in range(len(N)):
    print("the values of pi come out to be ",pi_l[i],"for throwing of points , N =",N[i])
plt.hist(pi_l,bins=10,edgecolor ="black")
plt.title("pi values")
plt.xlabel("pi values")
plt.ylabel("frequency of pi values occurred")
plt.show()

exp_pRNG_values = z.exponentially_pRNG(0.8,5000,1)
z.correlation_plot(exp_pRNG_values,5)
