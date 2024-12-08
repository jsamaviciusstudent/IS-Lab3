'''

    Create a Radial Basis Function Networks for approximation with:

    single input;
    single output;
    two Gaussian radial basis functions: F = exp(-(x-c)^2/(2*r^2)).

    Train the RBF network for approximation task for use 20 examples:

    input values are generated using equation x = 0.1: 1/22: 1;
    desired output values are calculated using formula y = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x)) / 2;
    select manually c1, r1 for the first RBF and c2, r2 for the second RBF;
    use perceptron training algorithm for estimating output layer parameters w1, w2, w0.

'''


'''
        b1
    w1(1) o w1(2)
        b2          b4(2)
x   w2(1) o w2(2)    o -                    -  Y
        b3          (predicted outcome)
    w3(1) o w3(2)

    6w + 4b = 10 weights to train

    '''

# Single input / output
# hidden layer is radial basis function

'''
    c1 r1
    R(radius func)  w1         b1
Y <                 >  o(activation) - Y
    R_2(radius)     w2
    c2 r2
'''

# Manually set first data points (C1,y1 ) = (x1,(x0-x1)) (C2,y2) = (X2,x3-x2))

# v = F_1 * w1 + f2 * w2 + b1

# 1. Create radial basis function network for approximation

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


learning_rate = 0.15
## Input
# 20 Input vectors, range form 0.1 to 1
X = np.arange(0.1, 1.01, 1/22)

# Compute d based on the given formula
d = ((1 + 0.6 * np.sin(2 * np.pi * X / 0.7)) + 0.3 * np.sin(2 * np.pi * X)) / 2

# Plot with black 'x' markers
plt.figure(figsize=(10, 6))
plt.plot(X, d, 'kx')
plt.grid()

def RadialBasisFunc(x, c, r):
    rbf = np.exp(-(x-c)**2/(2*r**2))
    return rbf
## c and r are manually set
# First radius
c1 =  X[2]
r1 = X[4] - X[2]
R1 = RadialBasisFunc(X, c1, r1)

# Second radius
c2 = X[17]
r2 = X[19] - X[17]
R2 = RadialBasisFunc(X, c2, r2)


# Setting random weights
w1 = np.random.randn()
w2 = np.random.randn()
w0 = np.random.randn()

Y = [0] * len(X)
# Estimating the output of the model
for epoch in range(1, 1000):
    for i in range(len(X)):

            R1 = RadialBasisFunc(X[i], c1, r1)
            R2 = RadialBasisFunc(X[i], c2, r2)

            y = R1 * w1 + R2 * w2 + w0
            Y[i] = y

            # Error calc:
            e = d[i] - Y[i]

            # Updating parameters:
            w1 = w1 + learning_rate * e * R1
            w2 = w2 + learning_rate * e * R2
            w0 = w0 + learning_rate * e


#print(Y)
plt.plot(X, Y, 'r-', label='RBF trained')
plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.legend()
plt.grid(True)
plt.show()
