import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset():
    np.random.seed(5)
    class_0 = np.random.normal(2,0.5,(50,3))
    class_1 = np.random.normal(5,0.5,(50,3))
    x=np.vstack((class_0 , class_1))
    y=np.array([0]*50 + [1]*50)
    return x,y
def split_dataset(x,y):
    return train_test_split(x,y,test_size=0.3)
def main():
    x,y = prepare_dataset()
    x_train ,x_test, y_train,y_test = split_dataset(x,y)
    print("Training Feature shape:", x_train.shape)
    print("Testing feature shape:",x_test.shape)
    print("Training Labels shape:",y_train.shape)
    print("Testing Labels Shape:" , y_test.shape)
if __name__=="__main__":
    main()
