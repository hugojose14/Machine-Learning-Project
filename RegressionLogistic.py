"@author: Hugo Jose"
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

with open ('transfusion.txt','r') as f:
    lines = f.readlines()
lines=[line.replace(' ', '') for line in lines]
with open('transfusion.txt', 'w') as f:
    f.writelines(lines)

dataSet = np.loadtxt("transfusion.txt", delimiter =",")

x=dataSet[:,:3]
y=dataSet[:,4]
scaler =MinMaxScaler()
normalize= scaler.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state =0,stratify=y)
c = [0.1, 0.01,0.001, 1, 0.0001, 0.0000001,0.000000001, 2.0, 2.1, 2.2, 3.0,3,1,3.2]
max_c = c[0]; max_score = 0
for i in c:
    clf = LogisticRegression(C=i, solver="lbfgs", max_iter = 1000000)
    sc = ((cross_val_score(clf, xtrain, ytrain, cv=10)).mean())
    if max_score < sc:
        max_score = sc
        max_c = i
print("-----------------------------------------")
print("{} | {}".format(max_score, max_c))
print("-----------------------------------------")

clf = LogisticRegression(C=max_c, solver="lbfgs", max_iter = 1000000).fit(xtrain, ytrain)
sc = ((cross_val_score(clf, xtrain, ytrain, cv=10)).mean())
pY = clf.predict(x[:,:])
fs = f1_score(y, pY)
recall = recall_score(y, pY)
precision = precision_score(y, pY)
acc = accuracy_score(y, pY)
tn, fp, fn, tp = confusion_matrix(y, pY).ravel()
print("-----------------------------------------")
print("\tPrecision: ", precision)
print("\tF1 Score: ",fs)
print("\tRecall: ",recall)
print("\tAccuracy: ", acc)
print("-----------------------------------------")
print("           Negative  Positive  ".format(tn, fp))
print("   Negative    {}        {}     ".format(tn, fp))
print("   Positive    {}        {}     ".format(fn, tp))




