### BEGIN PYTHON CODE ###
import cmath
import numpy
import math

# Cam Brown's HW2 Code

### Exercise A ###
def rect(r, phi):
    return r * (math.cos(phi) + math.sin(phi)*1j)

### Exercise B ###
def directSum(A, B):
    width = len(A[0]) + len(B[0])
    height = len(A) + len(B)
    C = numpy.zeros((height, width), dtype=numpy.array(0 + 0j).dtype)
    for row in range(height):
        for col in range(width):
            if row < len(A) and col < len(A[0]):
                C[row][col] = A[row][col]
            elif (row < len(B) + len(A) and row >= len(A)) and (col < len(A[0]) + len(B[0]) and col >= len(A[0])):
                C[row][col] = B[row-len(A)][col-len(A[0])]
    return C

def main():
    print(rect(2, math.pi))
    print(cmath.rect(2, math.pi))

    A = [[1, 1, 1], [2, 2, 2]]
    B = [[3, 3], [4, 4]]
    print(directSum(A, B))

if __name__ == '__main__':
    main()

### END PYTHON CODE ###
