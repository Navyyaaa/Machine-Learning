import numpy as np
import matplotlib.pyplot as plt

def compute_mean(data):
    return sum(data) / len(data)
def compute_variance(data):
    mean_val = compute_mean(data)
    return sum((x-mean_val) ** 2 for x in data) / len(data)
def histogram_data(feature , bins):
    hist, bin_edges = np.histogram(feature,bins=bins)
    return hist, bin_edges
def main():
    np.random.seed(2)
    data = np.random.normal(4,1,(100,3))
    feature = data[:,0]
    mean_val=compute_mean(feature)
    variance_val = compute_variance(feature)
    hist , bins = histogram_data(feature , 10)
    print("Mean:",mean_val)
    print("Variance:",variance_val)
    print("Histogram Values:",hist)
    print("Bin Ranges:" , bins)

    plt.hist(feature , bins=10)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title("histogram")
    plt.show()
if __name__ == "__main__":
    main()
