import numpy as np
import matplotlib.pyplot as plt
x = np.zeros(10)
for i in range(1, 11):
    x[i - 1] = i

y = np.zeros(10)
for i in range(1, 11):
    y[i - 1] = i*10

cost = np.zeros(1000)
W = np.arange(-50, 50, 0.1)

for i in range(len(W)):
    cost_value = 0.

    for index in range(10):
        H = W[i]*x[index] - y[index]
        cost_value = cost_value + pow(H, 2)/10

    cost[i] = cost_value

plt.plot(W, cost)
plt.xlabel('Weight')
plt.ylabel('Cost')
plt.show()

