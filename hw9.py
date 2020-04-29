# Cam Brown

# This module provides classical simulations of basic quantum algorithms (once you write them).

# In numpy, I think that the default complex dtype varies from platform to platform. If you want to explicitly use the default type in your code, use one.dtype (where one is defined just below).

import random
import math
import numpy as np

### CONSTANTS ###

# We haven't discussed this trivial case, but a 0-qbit state or gate is the complex scalar 1, represented as the following object. Notice that this object is neither the column vector numpy.array([1 + 0j]) nor the matrix numpy.array([[1 + 0j]]).
one = np.array(1 + 0j)

# Our favorite one-qbit states.
ket0 = np.array([1 + 0j, 0 + 0j])
ket1 = np.array([0 + 0j, 1 + 0j])
ketPlus = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
ketMinus = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])

# Our favorite one-qbit gates.
i = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])
x = np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
y = np.array([[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]])
z = np.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]])
h = np.array([[1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j],
              [1 / math.sqrt(2) + 0j, -1 / math.sqrt(2) + 0j]])

# Our favorite two-qbit gates.
cnot = np.array([[1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                 [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                 [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
                 [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]])
swap = np.array([[1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                 [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
                 [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                 [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j]])


### BIT STRINGS ###

# We represent an n-bit string --- that is, an element of {0, 1}^n --- in Python as a tuple of 0s and 1s.


def string(n, m):
    '''Converts a non-negative Python integer m to its corresponding bit string. As necessary, pads with leading 0s to bring the number of bits up to n.'''
    s = ()
    while m >= 1:
        s = (m % 2,) + s
        m = m // 2
    s = (n - len(s)) * (0,) + s
    return s


def integer(s):
    '''Converts a bit string s to its corresponding non-negative Python integer.'''
    m = 0
    for k in range(len(s)):
        m = 2 * m + s[k]
    return m


# returns the norm of a complex number
def norm(c):
    return math.sqrt(math.pow(c.real, 2) + math.pow(c.imag, 2))


def next(s):
    '''Given an n-bit string s, returns the next n-bit string. The order is lexicographic, except that there is a string after 1...1, namely 0...0.'''
    k = len(s) - 1
    while k >= 0 and s[k] == 1:
        k -= 1
    if k < 0:
        return len(s) * (0,)
    else:
        return s[:k] + (1,) + (len(s) - k - 1) * (0,)


def firstTest():
    # Constructs an unentangled two-qbit state |0> |psi> or |1> |psi>,
    # measures the first qbit, and then reconstructs the state.
    print("One should see 0s.")
    psi = uniform(1)
    state = tensor(ket0, psi)
    meas = first(state)
    print(state - tensor(meas[0], meas[1]))
    psi = uniform(1)
    state = tensor(ket1, psi)
    meas = first(state)
    print(state - tensor(meas[0], meas[1]))


def lastTest():
    # Constructs an unentangled two-qbit state |0> |psi> or |1> |psi>,
    # measures the first qbit, and then reconstructs the state.
    print("One should see 0s.")
    psi = uniform(1)
    state = tensor(ket0, psi)
    meas = last(state)
    print(state - tensor(meas[0], meas[1]))
    psi = uniform(1)
    state = tensor(ket1, psi)
    meas = last(state)
    print(state - tensor(meas[0], meas[1]))


def nextTest(n):
    '''A unit test for some basic bit-string routines. Should print the integers from 0 to 2^n - 1.'''
    s = string(n, 0)
    m = integer(s)
    print(m)
    s = next(string(n, m))
    while s != n * (0,):
        m = integer(s)
        print(m)
        s = next(string(n, m))


def addition(s, t):
    """Returns the mod-2 sum of two n-bit strings s and t."""
    return tuple([(s[i] + t[i]) % 2 for i in range(len(s))])


def dot(s, t):
    """Returns the mod-2 dot product of two n-bit strings s and t."""
    return sum([s[i] * t[i] for i in range(len(s))]) % 2


def reduction(a):
    """A is a list of m >= 1 bit strings of equal dimension n >= 1. In other words, A is a non-empty m x n binary 
    matrix. Returns the reduced row-echelon form of A. A itself is left unaltered. """
    b = a.copy()
    m = len(b)
    n = len(b[0])
    rank = 0
    for j in range(n):
        # Try to swap two rows to make b[rank, j] a leading 1.
        i = rank
        while i < m and b[i][j] == 0:
            i += 1
        if i != m:
            # Perform the swap.
            temp = b[i]
            b[i] = b[rank]
            b[rank] = temp
            # Reduce all leading 1s below the one we just made.
            for i in range(rank + 1, m):
                if b[i][j] == 1:
                    b[i] = addition(b[i], b[rank])
            rank += 1
    for j in range(n - 1, -1, -1):
        # Try to find the leading 1 in column j.
        i = m - 1
        while i >= 0 and b[i][j] != 1:
            i -= 1
        if i >= 0:
            # Use the leading 1 at b[i, j] to reduce 1s above it.
            for k in range(i):
                if b[k][j] == 1:
                    b[k] = addition(b[k], b[i])
    return b


def function(n, m, f):
    """Assumes that n = m = 1. The argument f is a Python function that takesas input an n-bit string alpha and 
    returns as output an m-bit string f(alpha). See deutschTest for examples of f. This function returns the (n + 
    m)-qbit gate F that corresponds to f. """
    if f((1,)) == (1,) and f((0,)) == (0,):
        return cnot
    elif f((1,)) == (0,) and f((0,)) == (1,):
        return np.dot(cnot, (tensor(i, x)))
    elif f((0,)) == (1,):
        return tensor(i, x)
    else:
        return tensor(i, i)


def deutsch(f):
    """Given a two-qbit gate representing a function f : {0, 1} -> {0, 1},outputs ket1 if f is constant and ket0 if f 
    is not constant. """
    return first(np.dot(tensor(h, h), np.dot(f, np.dot(tensor(h, h), tensor(ket1, ket1)))))[0]


def deutschTest():
    print("One should see ket0s")

    def f(x):
        return 1 - x[0],

    print(deutsch(function(1, 1, f)))

    def f(x):
        return x

    print(deutsch(function(1, 1, f)))
    print("One should see ket1s")

    def f(x):
        return (0,)

    print(deutsch(function(1, 1, f)))

    def f(x):
        return (1,)

    print(deutsch(function(1, 1, f)))


def application(gate, state):
    """Assumes n >= 1. Applies the n-qbit gate to the n-qbit state, returning
    an n-qbit state."""
    return np.dot(gate, state)


def tensor(a, b):
    """Assumes that a and b are both gates or a and b are both states. Let a be n-qbit and b be m-qbit, where n, 
    m >= 1. Returns the tensor product of a and b, which is (n + m)-qbit. """

    if len(np.shape(a)) == 1:  # if a is-a state
        tp = np.array([])
        for alpha in a:
            for beta in b:
                np.append(tp, [alpha * beta])

    else:
        for i in range(0, len(b)):  # traverses rows of b
            beta = np.dot(a, b[i][0])
            for j in range(1, len(b[i])):  # traverses columns of b
                alpha = np.dot(a, b[i][j])
                np.concatenate((beta, alpha), axis=1)

            if i != 0:
                np.concatenate((tp, beta), axis=0)
            else:
                tp = beta

    return tp


def first(state):
    """Assumes n >= 1. Given an n-qbit state, measures the first qbit. Returnsa pair (a tuple or list of two 
    elements) consisting of a classical one-qbit state (either ket0 or ket1) and an (n - 1)-qbit state. """
    sum_x =0
    for i in range(0, math.floor(len(state) / 2)):
        sum_x += math.pow(norm(state[i]), 2)
    x = math.sqrt(sum_x)
    y = math.sqrt(1 - math.pow(x, 2))

    measurement = random.uniform(0, 1)
    if measurement < math.pow(abs(x), 2):
        ketChi = (1 / x) * state[:math.floor(len(state) / 2)]
        answer = [ket0, ketChi]
    else:
        ketPhi = (1 / y) * state[math.floor(len(state) / 2):]
        answer = [ket1, ketPhi]
    return answer


def firstTest345(n, m):
    """Assumes n >= 1. Uses one more qbit than that, so that the total number of qbits is n + 1. The parameter m is
    how many tests to run. Should return a number close to 0.64 --- at least for large m. """
    psi0 = 3 / 5
    beta = uniform(n)
    psi1 = 4 / 5
    gamma = uniform(n)
    chi = psi0 * tensor(ket0, beta) + psi1 * tensor(ket1, gamma)


    def f():
        if (first(chi)[0] == ket0).all():
            return 0
        else:
            return 1

    acc = 0
    for i in range(m):
        acc += f()
    return acc / m


def last(state):
    """Assumes n>= 1. Given an n-qbit state, measures the last qbit. Returns
    a pair (a tuple or list of two elements) consisting of an (n-1)-qbit 
    state and a classical one-qbit state (either ket0 or ket1)"""
    sumx = 0
    for i in range(0, math.floor(len(state) / 2)):
        sumx += math.pow(norm(state[i]), 2)
    x = math.sqrt(sumx)
    y = math.sqrt(1 - math.pow(x, 2))

    measurement = random.uniform(0, 1)
    if measurement < math.pow(abs(x), 2):
        ketChi = (1 / x) * state[:1]
        answer = [ketChi, ket0]
    else:
        ketPhi = (1 / y) * state[2:]
        answer = [ketPhi, ket1]
    return answer


### MISCELLANY ###


def uniform(n):
    """Assumes n >= 0. Returns a uniformly random n-qbit state."""
    if n == 0:
        return one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = np.array(
                [random.normalvariate(0, 1) for i in range(2 ** n)])
            imags = np.array(
                [random.normalvariate(0, 1) for i in range(2 ** n)])
            psi = np.array([reals[i] + imags[i] * 1j for i in range(2 ** n)])
            psiNormSq = np.dot(np.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm


### MAIN ###


# It is conventional to have a main() function. Currently it does nothing. Change it to do whatever you want (or not).
def main():
    print("should be close to .64")
    result = firstTest345(2, 100000)
    print(result)
    if .63 < result < .65:
        print("it do be :)")
    else:
        print("it don't be :(")


# If the user imports this file into another program, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()
