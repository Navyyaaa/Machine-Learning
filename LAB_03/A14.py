import numpy as np
import matplotlib.pyplot as plt

def minkowski_distance(vector_a ,vector_b , p):
    total =0.0
    for i in range(len(vector_a)):
        total += abs(vector_a[i] - vector_b[i]) ** p
    return total ** (1 / p )
def main():
    np.random.seed(3)
    data = np.random.normal(5,1,(50,4))
    vector_1 = data[0]
    vector_2 = data[1]
    p_values = list(range(1, 11))
    distances = []
    for p in p_values:
        distances.append(minkowski_distance(vector_1 , vector_2 , p))
    print("p values:", p_values)
    print("Minkowski distances:",distances)
    plt.plot(p_values , distances ,marker='o')
    plt.xlabel("p values")
    plt.ylabel("Minkowski Distances")
    plt.title("Minikowski Distance vs p")
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()
