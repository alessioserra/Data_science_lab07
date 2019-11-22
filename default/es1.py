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
from matplotlib.pyplot import psd

"""FUNCTION"""
def preprocess1(X,n_partitions=40):
    output = []
    mean_tot = np.mean(X)
    std_tot = np.std(X)
    X = [ (el-mean_tot)/std_tot for el in X ] # Standardization
    l = len(X)
    for i in range(n_partitions):
        partition = X[int(i*l/n_partitions):int((i+1)*l/n_partitions)]
        output.append(np.mean(partition))
        output.append(np.std(partition))
        output.append(np.max(partition))
        output.append(np.min(partition))
    return np.array(output)

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
    
X_psd = []
X_stat = []
# Make matrix with statistic index
for x in X:
    X_psd.append(psd(x)[0]) # function for fourier spectrum
    X_stat.append(preprocess1(x))

clf = RandomForestClassifier()
f1 = cross_val_score(clf, X_psd, y, cv=5, scoring='f1_macro')
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

X_psd_eval = []
X_stat_eval = []
for keys in X_eval.keys():
    X_psd_eval.append(psd(X_eval[keys])[0])
    X_stat_eval.append(preprocess1(X_eval[keys])) 
    
clf_ev = RandomForestClassifier()
clf_ev.fit(X_psd, y)
assignments = clf_ev.predict(X_psd_eval)

dump_to_file("result.csv", assignments, X_eval)
print("Computed finished")