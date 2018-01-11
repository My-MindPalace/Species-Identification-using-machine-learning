from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xtrain,Ytrain)

def displayDataset (displayFlag):
    if displayFlag == True:
        print("Keys- {}".format(iris_dataset.keys()))
        print("Values: {}".format(iris_dataset.values()))
    print("\n\n\n")

def predictSpecies(Xnew):
    prediction = knn.predict(Xnew)
    print("The flower in consideration belongs to an Iris plant and has sepal length %s, sepal width %s, petal length %s and petal width %s"%(Xnew[0,0],Xnew[0,1],Xnew[0,2],Xnew[0,3]))
    print("It can belong to one of the four species which are 'setosa','versicolor','virginica'.")
    print("Prediction - {}".format(prediction))
    print("Predicted target name - {}".format(iris_dataset['target_names'][prediction]))
    print("\n\n\n")

def showAccuracy():
    test_prediction = knn.predict(Xtest)
    print("Test set prediction : {}\n".format(test_prediction))
    print("Test set predicted target names - {}".format(iris_dataset['target_names'][test_prediction]))
    print("Test set score - {:.2f}".format(np.mean(test_prediction == Ytest)))
    print("\n\n\n")

def main():
    Xnew = np.array([[5, 2.9, 1, 0.2]])
    choice = 0
    while (choice != 4):
        print("1.Show the dataset\n2.Show accuracy of the ML model\n3.Determine species of a given flower\n4.Exit")
        choice = int(input("Enter your choice - "))
        if (choice == 1):
            displayDataset(True)
        elif (choice == 2):
            showAccuracy()
        else:
            predictSpecies(Xnew)
if __name__=="__main__":
    main()
