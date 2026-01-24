import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def prepare_dataset():
    np.random.seed(6)
    class_0 = np.random.normal(2,0.5,(50,3))
    class_1 = np.random.normal(5 , 0.5,(50,3))
    x=np.vstack((class_0 , class_1))
    y=np.array([0]*50 + [1]*50)
    return x,y
def split_dataset(x,y):
    return train_test_split(x,y,test_size =0.3)
def train_knn(x_train,y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    return model
def main():
    x,y = prepare_dataset()
    x_train,x_test,y_train,y_test =split_dataset(x,y)
    knn_model = train_knn(x_train , y_train)
    print("Knn model trained with k = 3")
if __name__ =="__main__":
    main()
