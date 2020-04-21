# hw6
# Cam Brown
# April 18, 2020

import qc
import math

def check_error():
    errors = []
    deciles =[]
    iterations = 100000
    for i in range(0, iterations):
        state = qc.uniform(2)
        errors.append(state[0]*state[3] - state[1]*state[2])
    for i in range(0, 10):
        deciles.append(errors[math.floor(i*(iterations / 10))])
    print(deciles)

def main():
    check_error()

if __name__ == '__main__':
    main()
