import numpy as np
import matplotlib.pyplot as plt

# number of data points 
n = 5

# simple dataset
x = np.array([1,2,3,4,5])
y = np.array([5,8,11,14,17])

# Start with m =0 and c = 0

m = 0
c = 0

learning_rate = 0.01

for i in range(101):
    y_predicted = m * x + c
    
    #calculate the cost
    cost = (1/n) * sum([value ** 2 for value in(y - y_predicted)])
    plt.scatter(m, cost)
    
    #calculate gradients
    dm = -(2/n) * sum(x * (y - y_predicted))
    dc = -(2/n) * sum(y - y_predicted)
    
    #update the parameters
    m -= learning_rate * dm
    c -= learning_rate * dc
    
    print("m {}, c {}, cost{} iteration {}".format(m, c, cost, i))
plt.show()    
