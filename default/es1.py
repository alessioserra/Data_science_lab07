'''
Created on 20 nov 2019

@author: zierp
'''
import os
from scipy.io import wavfile as w
import numpy as np
from sklearn.model_selection import cross_val_score
import csv
from sklearn.ensemble.forest import RandomForestClassifier
from scipy.signal import welch # Fourier

"""FUNCTION"""
def preprocess(X):
    mean_tot = np.mean(X)
    std_tot = np.std(X)
    X = [ (el-mean_tot)/std_tot for el in X ] #Standardization
    return welch(X)[1]

def dump_to_file(filename, assignments, dataset):
    with open(filename, mode="w", newline="") as csvfile:
        
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ids, cluster in zip(dataset.keys(), assignments):
            writer.writerow({'Id': str(ids), 'Predicted': str(cluster)})
""""""

file_list = os.listdir("free-spoken-digit\\dev")
X = [] # features
y = [] # labels

for file in file_list:
    f = w.read('free-spoken-digit\\dev\\'+str(file))
    labels = str(file).split("_")
    label = labels[1].split(".")
    
    """Numpy array X"""
    X.append(preprocess(f[1])) #Standardization with method PreProcess
    """Label y"""
    y.append(int(label[0]))

clf = RandomForestClassifier(n_estimators=1000, n_jobs=2)
f1 = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
avg_f1 = np.mean(f1)
print("Accuracy: "+str(avg_f1))

# Evaluation
file_list = os.listdir("free-spoken-digit\\eval")

X_eval = {} # features for evaluation
for file in file_list:
    f = w.read('free-spoken-digit\\eval\\'+str(file), mmap=False)
    """Numpy array X"""
    filename = str(file).split(".")
    X_eval[str(filename[0])] = (preprocess(f[1])) #passing to floating points 

X_welch_eval = []
for keys in X_eval.keys():
    X_welch_eval.append(X_eval[keys])
    
clf_ev = RandomForestClassifier(n_estimators=1000, n_jobs=2)
clf_ev.fit(X, y)
assignments = clf_ev.predict(X_welch_eval)

dump_to_file("result.csv", assignments, X_eval)
print("Computing finished")