# Cam Brown

# This module provides classical simulations of basic quantum algorithms (once you write them).

# In numpy, I think that the default complex dtype varies from platform to platform. If you want to explicitly use the default type in your code, use one.dtype (where one is defined just below).

import random
import math
import cmath
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

# Our favorite three-qbit gates.
toffoli = np.array([[1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
                    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]])


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
    return tuple([(s[iter] + t[iter]) % 2 for iter in range(len(s))])


def dot(s, t):
    """Returns the mod-2 dot product of two n-bit strings s and t."""
    return sum([s[iter] * t[iter] for iter in range(len(s))]) % 2


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
            for iter in range(rank + 1, m):
                if b[iter][j] == 1:
                    b[iter] = addition(b[iter], b[rank])
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

    F = np.zeros((2 ** (n + m), 2 ** (n + m)), dtype=one.dtype)
    for a in range(0, 2 ** n):
        for b in range(0, 2 ** m):
            alpha = string(n, a)
            beta = string(m, b)
            beta_new = addition(beta, f(alpha))
            row_bits = alpha + beta_new
            col_bits = alpha + beta
            F[integer(row_bits)][integer(col_bits)] = 1 + 0j

    return F


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


# def tensor(a, b):
#     """Assumes that a and b are both gates or a and b are both states. Let a be n-qbit and b be m-qbit, where n,
#     m >= 1. Returns the tensor product of a and b, which is (n + m)-qbit. """

#     if len(a.shape) == 1:  # if a is-a state
#         tp = a[0] * b
#         for alpha in range(1, len(a)):
#             tp = np.concatenate((tp, a[alpha] * b), axis=0)

#     else:
#         for i in range(0, len(b)):  # traverses rows of b
#             beta = np.dot(a, b[i][0])
#             for j in range(1, len(b[i])):  # traverses columns of b
#                 alpha = np.dot(a, b[i][j])
#                 np.concatenate((beta, alpha), axis=1)

#             if i != 0:
#                 np.concatenate((tp, beta), axis=0)
#             else:
#                 tp = beta

#     return tp

def tensor(a, b):
    return np.kron(a, b)


def first(state):
    """Assumes n >= 1. Given an n-qbit state, measures the first qbit. Returnsa pair (a tuple or list of two
    elements) consisting of a classical one-qbit state (either ket0 or ket1) and an (n - 1)-qbit state. """
    sum_x = 0
    for iter in range(0, math.floor(len(state) / 2)):
        sum_x += math.pow(norm(state[iter]), 2)
    x = math.sqrt(sum_x)
    if x > 1: x = 1
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
    for iter in range(m):
        acc += f()
    return acc / m


def last(state):
    """Assumes n>= 1. Given an n-qbit state, measures the last qbit. Returns
    a pair (a tuple or list of two elements) consisting of an (n-1)-qbit
    state and a classical one-qbit state (either ket0 or ket1)"""
    sum_x = 0
    sum_y = 0
    for iter in range(0, len(state)):
        if i % 2 == 0:
            sum_x += math.pow(norm(state[iter]), 2)
        else:
            sum_y += math.pow(norm(state[iter]), 2)
    x = math.sqrt(sum_x)
    y = math.sqrt(sum_y)

    measurement = random.uniform(0, 1)
    if measurement < math.pow(abs(x), 2):
        ketChi = (1 / x) * state[::2]
        answer = [ketChi, ket0]
    else:
        ketPhi = (1 / y) * state[1::2]
        answer = [ketPhi, ket1]
    return answer


def lastTest345(n, m):
    """Assumes n >= 1. Uses one more qbit than that, so that the total number of qbits is n + 1. The parameter m is
    how many tests to run. Should return a number close to 0.64 --- at least for large m. """
    psi0 = 3 / 5
    beta = uniform(n)
    psi1 = 4 / 5
    gamma = uniform(n)
    chi = psi0 * tensor(beta, ket0) + psi1 * tensor(gamma, ket1)

    def f():
        if (first(chi)[0] == ket0).all():
            return 0
        else:
            return 1

    acc = 0
    for iter in range(m):
        acc += f()
    return acc / m


def power(stateOrGate, m):
    '''Given an n-qbit gate or state and m >= 1, returns the mth tensor power,
    which is an (n * m)-qbit gate or state. Assumes n >= 1. For the sake of
    time and memory, m should be small.'''
    tp = stateOrGate
    for iter in range(0, m - 1):
        tp = tensor(tp, stateOrGate)
    return tp


def bersteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate f representing a function {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown w in {0, 1}^n, returns the list or tuple of n classical one-qbit states (ket0 or ket1) corresponding to w.'''
    cp0 = tensor(power(ket0, n), ket1)
    cp1 = np.dot(cp0, power(h, n + 1))
    cp2 = np.dot(cp1, f)
    cp3 = np.dot(cp2, power(h, n + 1))
    part_meas = []
    part_meas.append(first(cp3)[0])
    rest = first(cp3)[1]

    for iter in range(1, n):
        part_meas.append(first(rest)[0])
        rest = first(rest)[1]

    gamma = ()
    for iter in range(0, len(part_meas)):
        if ((part_meas[i] == ket0).all()):
            gamma += (0,)
        else:
            gamma += (1,)

    return gamma


def randomBitString(n):
    ''' generates a random bit string of length n '''
    bit_str = string(n, random.randint(0, 2 ** n - 1))
    return bit_str


def bersteinVaziraniTest(n):
    delta = randomBitString(n)

    # function f that goes from {0,1}^n --> {0,1}
    def f(alpha):
        return (dot(delta, alpha),)

    F = function(n, 1, f)
    print("F is: ")
    print(F)
    bern_res = bersteinVazirani(n, F)
    print("\ndelta: ")
    print(delta)
    print("\nIf this thing matches delta we are g o l d e n")
    print(bern_res)


def simon(n, f):
    """ The inputs are an integer n >= and an (n + n - 1)-qbit gate f representing a function {0, 1}^n -> {0,
    1}^(n - 1) hiding an n-bit string w as in the Simon (1994) problem. Returns a list of n classical one-qbit states
    (ket0 or ket1) corresponding to a uniformly random bit string gamma that is perpendicular to w. """
    global i
    cp0 = power(ket0, 2 * n - 1)
    cp1 = tensor(power(h, n), power(i, n - 1))
    # print("cp0 shape: ")
    # print(cp0.shape)
    # print("cp1 shape: ")
    # print(cp1.shape)
    cp2 = application(cp1, cp0)
    # print("cp2 shape: ")
    # print(cp2.shape)
    cp3 = application(f, cp2)
    bot_meas = []
    bot_meas.append(last(cp3)[1])
    rem = last(cp3)[0]
    for iter in range(1, n - 1):
        bot_meas.append(last(rem)[1])
        rem = last(rem)[0]
    cp4 = np.dot(rem, power(h, n))
    top_meas = []
    top_meas.append(first(cp4)[0])
    rem = first(cp4)[1]
    for iter in range(1, n):
        top_meas.append(first(rem)[0])
        rem = first(rem)[1]
    return top_meas


def allZeros(gammas):
    for item in gammas[len(gammas) - 1]:
        if item != 0:
            return False
    return True


def simonTest(n):
    ''' function to do the stackin of the gammers '''

    def f(a):
        return a[1:]

    delta = (1,) + (n - 1) * (0,)
    F = function(n, n - 1, f)
    print(F.shape)
    s = simon(n, F)
    gammas = []
    while len(gammas) < n - 1:
        s = simon(n, F)
        gamma = ()
        for iter in range(0, len(s)):
            if ((s[i] == ket0).all()):
                gamma += (0,)
            else:
                gamma += (1,)
        print(gamma)
        gammas.append(gamma)
        gammas = reduction(gammas)
        if allZeros(gammas):
            gammas.pop()
    print(gammas)
    print(delta)
    for gamma in gammas:
        # hopefully prints out a bunch of zeros
        print(dot(gamma, delta))


def powerMod(k, l, m):
    """ Given non-negative integer k, non-negative integer l, and positive
        integer m. Computes k^l mod m. Returns an integer in {0, ..., m - 1}.  """
    if l == 0 and m != 1:
        return 1
    elif l == 0 and m == 1:
        return 0

    gPow = math.ceil(math.log(l, 2) + 1)
    str_l = string(gPow, l)
    k_ttl = 1
    k_pow = k
    for iter in range(0, gPow):
        if str_l[gPow - iter - 1] == 1:
            k_ttl *= k_pow
            k_ttl = k_ttl % m
        k_pow = k_pow ** 2
        k_pow = k_pow % m

    return k_ttl


def fourier(n):
    """ Returns the n-qbit quantum Fourier transform gate T. """
    coeff_T = (1 / (2 ** (n / 2)))
    T = np.zeros([2 ** n, 2 ** n], dtype=one.dtype)
    for row in range(0, 2 ** n):
        for col in range(0, 2 ** n):
            T[row][col] = cmath.rect(coeff_T, 2 * math.pi * col * (row / (2 ** n)))
    return T


def shor(n, f):
    """ Assumes n >= 1. Given an (n + n)-qbit gate f representing a function
        f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list of
        classical one-qbit states (ket0 or ket1) corresponding to an n-bit string
        that satisfies certain mathematical properties. """
    global i
    cp0 = power(ket0, 2 * n)
    cp1 = tensor(power(h, n), power(i, n))
    # print("cp0 shape: ")
    # print(cp0.shape)
    # print("cp1 shape: ")
    # print(cp1.shape)
    cp2 = application(cp1, cp0)
    # print("cp2 shape: ")
    # print(cp2.shape)
    cp3 = application(f, cp2)
    bot_meas = []
    bot_meas.append(last(cp3)[1])
    rem = last(cp3)[0]
    for iter in range(1, n):
        bot_meas.append(last(rem)[1])
        rem = last(rem)[0]
    # cp4 = np.dot(rem, power(h, n))
    cp5 = application(fourier(n), rem)
    top_meas = [first(cp5)[0]]
    rem = first(cp5)[1]
    for iter in range(1, n):
        top_meas.append(first(rem)[0])
        rem = first(rem)[1]
    return top_meas


def shorTest_old(n, m):
    k = m
    while math.gcd(k, m) != 1:
        k = random.randint(2, m)

    def f(a):
        int_a = integer(a)
        pow_a = powerMod(k, int_a, m)
        str_a = string(n, pow_a)
        return str_a

    # power_mods = []
    # i = 1
    # power_mods.append(powerMod(k, 0, m))
    # while powerMod(k, i, m) != power_mods[len(power_mods)]:
    #     power_mods.append(powerMod(k, i, m))
    #     i += 1
    # return power_mods

    F = function(n, n, f)

    # print(shor(n, F))


def shorTest(n, m):
    k = m
    while math.gcd(k, m) != 1:
        k = random.randint(2, m)

    return period_finding(n, k, m)


def fraction(x0, j):
    if x0 == 0 or j < 0:
        return (0, 1)
    else:
        a0 = math.floor(1 / x0)
        x1 = 1 / x0 - a0
        if fraction(x1, j - 1) == 0:
            return (1, a0)
        else:
            return (fraction(x1, j - 1)[1],
                    a0 * fraction(x1, j - 1)[1] + fraction(x1, j - 1)[0])


def continuedFraction(n, m, x0):
    j = 0
    while True:
        answer = fraction(x0, j)
        if answer == 0:
            c = 0
            d = 1
        else:
            c = answer[0]
            d = answer[1]
        if abs(x0 - c / d) <= 1 / math.pow(2, (n + 1)) or d >= m:
            return c, d
        else:
            j += 1


def ket_to_int(arr):
    s = ()
    for item in arr:
        if (item == ket0).all():
            s += (0,)
        elif (item == ket1).all():
            s += (1,)
    return integer(s)


def ket_to_str(arr):
    s = ()
    for item in arr:
        if (item == ket0).all():
            s += (0,)
        elif (item == ket1).all():
            s += (1,)
    return s


def period_finding(n, k, m):
    # step 1 implied by input params
    # step 2 done in shorTest

    def f(a):
        int_a = integer(a)
        pow_a = powerMod(k, int_a, m)
        str_a = string(n, pow_a)
        return str_a

    F = function(n, n, f)

    # step 3
    while True:
        d = m
        d_prime = m

        # step B
        while d >= m:
            b = shor(n, F)
            b = ket_to_int(b)
            d = continuedFraction(n, m, b / 2 ** n)[1]

        # step C
        if powerMod(k, d, m) == 1:
            p = d
            return p

        # step D
        while d_prime >= m:
            b = shor(n, F)
            b = ket_to_int(b)
            d_prime = continuedFraction(n, m, b / 2 ** n)[1]

        # step E
        if powerMod(k, d_prime, m) == 1:
            p = d_prime
            return p

        # step F
        lcm = (d * d_prime) / math.gcd(d, d_prime)

        # step G
        if powerMod(k, lcm, m) == 1:
            p = lcm
            return p


def grover(n, f):
    t = math.asin(1 / (2 ** (n / 2)))
    l = round((math.pi / (4 * t)) - 0.5)  # number of times we're gonna fire the loop
    ket_rho = power(ketPlus, n)
    ket_bra_rho = np.outer(ket_rho, ket_rho)
    i_n = power(i, n)
    R = (2 * ket_bra_rho) - i_n

    base_state = tensor(power(ket0, n), ket1)
    base_state = application(power(h, n + 1), base_state)
    for counter in range(0, l):
        base_state = application(f, base_state)
        base_state = application(tensor(R, i), base_state)

    top_meas = [first(base_state)[0]]
    rem = first(base_state)[1]
    for j in range(1, n):
        top_meas.append(first(rem)[0])
        rem = first(rem)[1]

    temp = ket_to_int(top_meas)
    return string(temp, n)


def groverTest(n):
    delta = random.randint(0, 2 ** n)
    str_delta = string(delta, n)

    def f(a):
        if a == str_delta:
            return 1,
        return 0,

    F = function(n, 1, f)
    grover_delta = grover(n, F)

    print("true delta: " + str(str_delta))
    print("grover delta: " + str(grover_delta))


def control(gate):
    ctrl = np.zeros((2 * len(gate), 2 * len(gate)), dtype=one.dtype)
    for j in range(0, 2 * len(gate)):
        for k in range(0, 2 * len(gate)):
            if j == k and j < len(gate) and k < len(gate):
                ctrl[j][k] = 1
            elif j >= len(gate) and k >= len(gate):
                ctrl[j][k] = gate[j - len(gate)][k - len(gate)]
    return ctrl


### extras.py ###


def distance(gate, m):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is n-qbit), performs swaps to insert m extra wires between the first qbit and the other n qbits. Returns an (n + 1 + m)-qbit gate.'''
    n = round(math.log2(len(gate))) - 1
    if n == 1:
        swapper = swap
    else:
        swapper = tensor(swap, power(i, n - 1))
    u = gate.copy()
    for k in range(m):
        u = tensor(i, u)
        swapper = tensor(swapper, i)
        u = np.matmul(swapper, np.matmul(u, swapper))
    return u


# The six crucial gates, which are also observables.
m07 = tensor(x, tensor(x, tensor(x, tensor(i, tensor(i, tensor(i, x))))))
m17 = tensor(x, tensor(x, tensor(i, tensor(x, tensor(i, tensor(x, i))))))
m27 = tensor(x, tensor(i, tensor(x, tensor(x, tensor(x, tensor(i, i))))))
n07 = tensor(z, tensor(z, tensor(z, tensor(i, tensor(i, tensor(i, z))))))
n17 = tensor(z, tensor(z, tensor(i, tensor(z, tensor(i, tensor(z, i))))))
n27 = tensor(z, tensor(i, tensor(z, tensor(z, tensor(z, tensor(i, i))))))

# Encoded gates except for CNOT.
i7 = power(i, 7)
x7 = power(x, 7)
z7 = power(z, 7)
h7 = power(h, 7)


# Code to check that the encoded states are orthonormal.
# print(numpy.dot(numpy.conj(ket07), ket07))
# print(numpy.dot(numpy.conj(ket07), ket17))
# print(numpy.dot(numpy.conj(ket17), ket17))

def error7(state7):
    '''Applies one of the 22 errors (including the trivial error) to state7, and returns the corrupted state.'''
    index = random.randrange(23)
    if index < 7:
        j = index
        error = tensor(power(i, 6 - j), tensor(x, power(i, j)))
        return application(error, state7)
    elif index < 14:
        j = index - 7
        error = tensor(power(i, 6 - j), tensor(y, power(i, j)))
        return application(error, state7)
    elif index < 21:
        j = index - 14
        error = tensor(power(i, 6 - j), tensor(z, power(i, j)))
        return application(error, state7)
    else:
        return state7


### constants ###

L = np.matmul(((power(i, 7)) + m07) / 2, np.matmul(((power(i, 7)) + m17) / 2, ((power(i, 7)) + m27) / 2))

e7 = (2 ** (3 / 2)) * L

ket07 = np.matmul(e7, power(ket0, 7))
ket17 = np.matmul(e7, power(ket1, 7))


def detection(state7):

    #initial state
    state = tensor(power(ket0, 6), state7)

    # apply first hadamard layer
    state = application(tensor(power(h, 6), power(i, 7)), state)

    # each control step
    state = application((tensor( power(i, 5), distance( control(n07), 0))), state)
    state = application((tensor( power(i, 4), distance( control(n17), 1))), state)
    state = application((tensor( power(i, 3), distance( control(n27), 2))), state)
    state = application((tensor(power(i, 2), distance(control(m07), 3))), state)
    state = application((tensor(power(i, 1), distance(control(m17), 4))), state)
    state = application( distance(control(m27), 5), state)

    # second layer of hadamards
    state = application( tensor(power(h, 6), power(i, 7)), state)

    str = ()

    for j in range(0, 6):
        first_qbit = first(state)
        if (first_qbit == ket0).all():
            str += (0,)
        elif (first_qbit == ket1).all():
            str += (1,)
        state = state[1:]

    if len(str) != 6:
        print("detection failed: string to short!")
        exit(1)

    return str

def decoding(state7):
    if state7 == ket07:
        return ket0
    elif state7 == ket17:
        return ket1
    print("both decoding cases failed, exiting")
    exit(1)

def correction7(diagnosis):
    '''Given the output from detection7, returns the uncorrupted seven-qbit state.'''
    # !! Replace the following line with your equivalent code for converting ket0s to 0s and ket1s to 1s.
    syndrome = ket_to_str(diagnosis)
    if syndrome[0:3] == (0, 0, 0):
        # No Z error has occurred. Worry about only X errors.
        if syndrome[3:6] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(x, power(i, 6)))
        elif syndrome[3:6] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(x, power(i, 5)))
        elif syndrome[3:6] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(x, power(i, 4)))
        elif syndrome[3:6] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(x, power(i, 3)))
        elif syndrome[3:6] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(x, power(i, 2)))
        elif syndrome[3:6] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(x, power(i, 1)))
        elif syndrome[3:6] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(x, power(i, 0)))
        else:
            correction = power(i, 7)
    elif syndrome[3:6] == (0, 0, 0):
        # No X error has occurred. Worry about only Z errors.
        if syndrome[0:3] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(z, power(i, 6)))
        elif syndrome[0:3] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(z, power(i, 5)))
        elif syndrome[0:3] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(z, power(i, 4)))
        elif syndrome[0:3] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(z, power(i, 3)))
        elif syndrome[0:3] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(z, power(i, 2)))
        elif syndrome[0:3] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(z, power(i, 1)))
        elif syndrome[0:3] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(z, power(i, 0)))
        else:
            correction = power(i, 7)
    else:
        # A Y error has occurred.
        if syndrome[0:3] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(y, power(i, 6)))
        elif syndrome[0:3] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(y, power(i, 5)))
        elif syndrome[0:3] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(y, power(i, 4)))
        elif syndrome[0:3] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(y, power(i, 3)))
        elif syndrome[0:3] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(y, power(i, 2)))
        elif syndrome[0:3] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(y, power(i, 1)))
        elif syndrome[0:3] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(y, power(i, 0)))
        else:
            correction = power(i, 7)
    return application(correction, diagnosis[-1])


### MISCELLANY ###


def uniform(n):
    """Assumes n >= 0. Returns a uniformly random n-qbit state."""
    if n == 0:
        return one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = np.array(
                [random.normalvariate(0, 1) for iter in range(2 ** n)])
            imags = np.array(
                [random.normalvariate(0, 1) for iter in range(2 ** n)])
            psi = np.array([reals[iter] + imags[i] * 1j for iter in range(2 ** n)])
            psiNormSq = np.dot(np.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm


### MAIN ###


# It is conventional to have a main() function. Currently it does nothing. Change it to do whatever you want (or not).
def main():
     print(detection(ket07))


# If the user imports this file into another program, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()
