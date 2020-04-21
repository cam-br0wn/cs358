# Cam Brown

# This module provides classical simulations of basic quantum algorithms (once you write them).

# In numpy, I think that the default complex dtype varies from platform to platform. If you want to explicitly use the default type in your code, use one.dtype (where one is defined just below).

import random
import math
import cmath
import numpy



### CONSTANTS ###

# We haven't discussed this trivial case, but a 0-qbit state or gate is the complex scalar 1, represented as the following object. Notice that this object is neither the column vector numpy.array([1 + 0j]) nor the matrix numpy.array([[1 + 0j]]).
one = numpy.array(1 + 0j)

# Our favorite one-qbit states.
ket0 = numpy.array([1 + 0j, 0 + 0j])
ket1 = numpy.array([0 + 0j, 1 + 0j])
ketPlus = numpy.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
ketMinus = numpy.array([1 / math.sqrt(2), -1 / math.sqrt(2)])

# Our favorite one-qbit gates.
i = numpy.array([
    [1 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j]])
x = numpy.array([
    [0 + 0j, 1 + 0j],
    [1 + 0j, 0 + 0j]])
y = numpy.array([
    [0 + 0j, 0 - 1j],
    [0 + 1j, 0 + 0j]])
z = numpy.array([
    [1 + 0j, 0 + 0j],
    [0 + 0j, -1 + 0j]])
h = numpy.array([
    [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j],
    [1 / math.sqrt(2) + 0j, -1 / math.sqrt(2) + 0j]])

# Our favorite two-qbit gates.
cnot = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j], 
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]]) 
swap = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j], 
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

# returns the tensor product of two one-qbit states
def tensor(ketChi, ketPsi):
  print(ketPsi[0])
  print(ketPsi[1])
  tense = numpy.array(
    [ketChi[0] * ketPsi[0],
    ketChi[0] * ketPsi[1],
    ketChi[1] * ketPsi[0],
    ketChi[1] * ketPsi[1]])
  return tense

def first(s):
  ## |psi> = sigma*|0> tensor'd |chi> + tau*|1> tensor'd |phi>
  print(s)
  x = math.sqrt(math.pow(norm(s[0]), 2) + math.pow(norm(s[1]), 2))
  y = math.sqrt(math.pow(norm(s[2]), 2) + math.pow(norm(s[3]), 2))
    
  # ketPsi = tensor(x*ket0, ketChi) + tensor(y*ket1, ketPhi)

  measurement = random.uniform(0, 1)
  if(measurement < math.pow(abs(x), 2)):
    ketChi = (1 / x) * s[:1]
    answer = [ket0, ketChi]
  else:
    ketPhi = (1 / y) * s[2:]
    answer = [ket1, ketPhi]
  return answer
  

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
  print(state)
  meas = first(state)
  print(state - tensor(meas[0], meas[1]))
  psi = uniform(1)
  state = tensor(ket1, psi)
  meas = first(state)
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
    '''Returns the mod-2 sum of two n-bit strings s and t.'''
    return tuple([(s[i] + t[i]) % 2 for i in range(len(s))])


def dot(s, t):
    '''Returns the mod-2 dot product of two n-bit strings s and t.'''
    return sum([s[i] * t[i] for i in range(len(s))]) % 2

def reduction(a):
    '''A is a list of m >= 1 bit strings of equal dimension n >= 1. In other words, A is a non-empty m x n binary matrix. Returns the reduced row-echelon form of A. A itself is left unaltered.'''
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


### MISCELLANY ###

def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            psi = numpy.array([reals[i] + imags[i] * 1j for i in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm



### MAIN ###

# It is conventional to have a main() function. Currently it does nothing. Change it to do whatever you want (or not).
def main():
    firstTest()

# If the user imports this file into another program, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()

 