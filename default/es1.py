'''
Created on 20 nov 2019

@author: zierp
'''
import os
from scipy.io import wavfile as w
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import csv
from sklearn.ensemble.forest import RandomForestClassifier

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
    X.append(f[1].astype(np.float32)) #passing to floating points
    """Label y"""
    y.append(int(label[0]))
    

X_stat = []

# Make matrix with statistic index

for x in X:
    row = np.array(x)
    mean = (sum(x)/len(x))
    var = np.var(row)
    maximumm = max(x)
    minimumm = min(x)
    rows = [mean, var, maximumm, minimumm]
    X_stat.append(rows)  

clf = RandomForestClassifier()
f1 = cross_val_score(clf, X_stat, y, cv=5, scoring='f1_macro')
avg_f1 = np.mean(f1)
print("Accuracy: "+str(avg_f1))

# Evaluation
file_list = os.listdir("free-spoken-digit\\eval")

X_eval = {} # features for evaluation
for file in file_list:
    f = w.read('free-spoken-digit\\eval\\'+str(file), mmap=False)
    """Numpy array X"""
    filename = str(file).split(".")
    X_eval[str(filename[0])] = (f[1].astype(np.float32)) #passing to floating points 

X_stat_eval = []
for keys in X_eval.keys():
        
    for x in X_eval[keys]:
        row = np.array(x)
        mean = np.mean(x)
        var = np.var(row)
        maximumm = np.max(x)
        minimumm = np.min(x)
        rows = [mean, var, maximumm, minimumm]
        X_stat_eval.append(rows) 
    
clf_ev = RandomForestClassifier()
clf_ev.fit(X_stat, y)
assignments = clf_ev.predict(X_stat_eval)

dump_to_file("result.csv", assignments, X_eval)
print("Computed finished")