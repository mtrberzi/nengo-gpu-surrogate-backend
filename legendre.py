#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# Legendre polynomial of degree n.
# The closed-form representation is
# P_n(x) = 2^n * (sum from k=0 to n) x^k * (n choose k)*([n+k+-1]/2 choose n) 
def legendre(n):
    coeffs = []
    for k in range(n+1):
        c1 = scipy.special.binom(n, k)
        c2 = scipy.special.binom((n+k-1)/2, n)
        coeffs.append(c1 * c2)
    def L(x):
        accumulator = np.zeros(x.shape)
        argument = np.ones(x.shape)
        for k in range(n+1):
            accumulator += np.multiply(argument, coeffs[k])
            argument = np.multiply(argument, x)
        return 2.0**n * accumulator
    return L

x = np.linspace(-1.0, 1.0, 100)

for n in range(6):
    L = legendre(n)
    y = L(x)
    plt.plot(x, y)
plt.show()
