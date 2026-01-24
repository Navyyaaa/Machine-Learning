import numpy as np
from scipy.spatial.distance import minkowski

def custom_minkowski_distance(vector_a , vector_b , p):
    total = 0.0
    for i in range(len(vector_a)):
        total +=abs(vector_a[i] - vector_b[i]) ** p
    return total ** (1 / p)

def main():
    np.random.seed(4)
    data = np.random.normal(3,1,(20,5))
    vector_1 = data[0]
    vector_2 = data[1]
    p_value = 3
    custom_distance = custom_minkowski_distance(vector_1 ,vector_2, p_value)
    scipy_distance = minkowski(vector_1 ,vector_2, p_value)
    print("Custom Minkowski Distance:", custom_distance)
    print("Scipy Minkowski distance:" , scipy_distance)
if __name__ == "__main__":
    main()
