import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def dot_product(v1,v2):
    result =0.0
    for i in range(len(v1)):
        result +=v1[i] *v2[i]
        return result
def euclidean_norm(vec):
    sum_sq =0.0
    for val in vec:
        sum_sq +=val *val
    return sum_sq **0.5
def main():
    A=np.array([2,3,5])
    B=np.array([5,6,7])
    custom_dot= dot_product(A,B)
    custom_norm =euclidean_norm(A)
    numpy_dot = np.dot(A,B)
    numpy_norm = np.linalg.norm(A)
    print("Dot Product:", custom_dot)
    print("Numpy Dot Product:" , numpy_dot)
    print("Euclidean Norm:",custom_norm)
    print("Numpy_Euclidean Norm:", numpy_norm)
if __name__ == "__main__":
    main()
