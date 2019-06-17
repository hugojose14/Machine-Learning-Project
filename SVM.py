"@author: Hugo Jose"
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metricas
from sklearn import svm
from sklearn.metrics import confusion_matrix
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#load dataSet breast cancer
cancer = load_breast_cancer()
x= cancer.data
y= cancer.target

xtrain, xtest, ytrain, ytest = train_test_split(cancer.data, cancer.target, test_size = 0.30, random_state =0,stratify=cancer.target)

clf = svm.SVC(max_iter=100).fit(xtrain,ytrain)

modelo = clf.predict(xtest)

f1_score= metricas.f1_score(ytest,modelo)

recall = metricas.recall_score(ytest,modelo)

accuracy = metricas.accuracy_score(ytest,modelo)

precision = metricas.precision_score(ytest,modelo)

true_negative, false_positive, false_negative, true_positive = metricas.confusion_matrix(ytest,modelo).ravel()
print("-------------------------------")
print(" Matriz de confusion\n")
print("",[true_negative,false_positive])
print("",[false_negative,true_positive])
print("-------------------------------")
print("\nrecall",recall)
print("-------------------------------")
print("precision",precision)
print("-------------------------------")
print("accuracy",accuracy)
print("-------------------------------")

"""
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

interact(plot_3D, elev=[-90, 90], azip=(-180, 180),
         X=fixed(X), y=fixed(y));
"""