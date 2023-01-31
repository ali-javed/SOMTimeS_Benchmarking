
#import libraries
from pathlib import Path
import time
from SelfOrganizingMap import SelfOrganizingMap
import numpy as np
from pyts import datasets
import csv
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa
import warnings
warnings.filterwarnings("ignore")

########################################
#helper functions in this block
#rand index calculation function

def rand_index_score(clusters, labels):
    ####################
    # Function returns the Rand Index score between two datasets
    # inputs:
    # clusters -- cluster labels
    # labels -- class labels (or ground truth)
    ###########################
    # output: Rand Index
    ####################

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(labels), 2).sum()
    A = np.c_[(clusters, labels)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

#calculate different clustering accuracy metrics
def evaluate(classes, clusters):
    ####################
    # Function calculates all six metrics used in paper
    # inputs:
    # clusters -- cluster labels
    # clusters -- class labels (or ground truth)
    # scores --- scores dictionary containing scores
    ###########################
    # output: all 6 metrics calculated
    ####################
    scoresDict = {}
    scoresDict['adjusted Mutual information'] = adjusted_mutual_info_score(classes, clusters)
    scoresDict['adjusted Rand index'] = adjusted_rand_score(classes, clusters)
    scoresDict['Homogeneity'] = metrics.homogeneity_score(classes, clusters)
    scoresDict['Completeness'] = metrics.completeness_score(classes, clusters)
    scoresDict['Fowlkes Mallows'] = metrics.fowlkes_mallows_score(classes, clusters)
    scoresDict['Rand index'] = rand_index_score(classes, clusters)

    return scoresDict

################################################################################
np.random.seed(0)

# create output file to save results
outputPath = 'scores1.csv'
fieldNames = ['dataset', 'time', 'Calculated', 'totalCalls', 'Percent Pruned', 'dtwCallsPerEpoch', 'timePerEpoch', 'n', 'slength','adjusted Mutual information','adjusted Rand index','Homogeneity','Completeness','Fowlkes Mallows','Rand index']

fout = open(outputPath, "w")
csvw = csv.DictWriter(fout, fieldnames=fieldNames)
csvw.writeheader()
fout.close()

#window size for dtw between 0 and 1
wSize = 0.05
epochs =10

# Pick and compare performance on any dataset
UCR_DataSetNames = datasets.ucr_dataset_list()

# loop through datasets
counter = 0
for datasetName in UCR_DataSetNames[counter:]:
    try:
        obj = datasets.fetch_ucr_dataset(dataset=datasetName)
    except:
        continue

    inputs = np.append(obj['data_train'], obj['data_test'], axis=0)
    labels = np.append(obj['target_train'], obj['target_test'])
    # replace nan with zero
    inputs = np.nan_to_num(inputs, nan=0)

    classes = np.asarray(labels)
    k = len(set(classes))
    hiddenSize = [1, k]

    if -1 in classes:
        continue
    if datasets.ucr_dataset_info(datasetName)['n_timestamps' ]== 'Varying' or datasets.ucr_dataset_info(datasetName)['n_timestamps' ] == 'Variable':
        continue
    #clustering 1 class is trivial and not used in benchmark with known value for k
    if k== 1:
        continue

    if min(classes)>0:
        classes = classes - min(classes)
    classes = np.asarray(classes)

    print('Creating SOM... ')
    print(str(counter)+'. '+datasetName)
    counter+=1

    print('Hidden Size is: '+str(hiddenSize))
    SOM = SelfOrganizingMap(inputSize = len(inputs[0]), hiddenSize = hiddenSize)

    windowSize = int(len(inputs[1]) * wSize)

    start_time = time.time()
    stats = SOM.iterate(inputs,epochs = epochs,windowSize = windowSize,k=k,randomInitilization=False)

    stats['time'] = time.time() - start_time
    scores = evaluate(classes,stats['labels'])
    stats['dataset'] = datasetName
    #total calls for SOM are the number of inputs times k (if the mesh size is k), times epochs
    stats['totalCalls'] = (len(inputs)) * k * epochs
    stats['Percent Pruned'] = (1 - (stats['Calculated']/stats['totalCalls']))*100
    stats['n'] = len(inputs)
    stats['slength'] = datasets.ucr_dataset_info(datasetName)['n_timestamps' ]
    for measure in scores:
        stats[measure] = scores[measure]

    del stats['labels']
    fout = open(outputPath, "a")
    csvw = csv.DictWriter(fout, fieldnames=fieldNames)
    csvw.writerow(stats)
    fout.close()