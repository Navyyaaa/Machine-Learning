import numpy as np
def compute_mean(data):
    return sum(data) / len(data)
def compute_variance(data):
    mean_val=compute_mean(data)
    return sum((x-mean_val) ** 2 for x in data) / len(data)
def compute_std(data):
    return compute_variance(data) **0.5

def dataset_statistics(matrix):
    means =[]
    variances=[]
    stds = []
    for col in range(matrix.shape[1]):
        column = matrix[: ,col]
        means.append(compute_mean(column))
        variances.append(compute_variance(column))
        stds.append(compute_std(column))
    return np.array(means) ,np.array(variances) ,np.array(stds)
def main():
    np.random.seed(1)
    class_1 = np.random.normal(2 , 0.6, (40 , 2))
    class_2 =np.random.normal(5 , 0.6 ,(40,2))
    centroid_1 = class_1.mean(axis= 0)
    centroid_2 = class_2.mean(axis=0)
    spread_1 = np.std(class_1 ,axis=0)
    spread_2 = np.std(class_2 ,axis=0)
    interclass_distance = np.linalg.norm(centroid_1 - centroid_2)
    mean_1 ,var_1,std_1 =dataset_statistics(class_1)
    mean_2 ,var_2,std_2 =dataset_statistics(class_2)

    print("class 1 cetroid:",centroid_1)
    print("class 2 centroid:",centroid_2)
    print("class 1 spread:",spread_1)
    print("class 2 spread:",spread_2)
    print("interclass distance:",interclass_distance)
    print("class 1 feature_wise mean:",mean_1)
    print("class 1 feature_wise variance:",var_1)
    print("class 1 feature_wise std Dev:",std_1)
    print("class 2 feature-wise Mean:",mean_2)
    print("class 2 feature-wise variance:",var_2)
    print("class 2 feature-wise Std Dev:",std_2)
if __name__ =="__main__":
    main()

