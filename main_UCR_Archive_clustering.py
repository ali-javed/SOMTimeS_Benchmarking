
##############################
# author: Ali Javed
# December 17 2022
# email: ajaved@stanford.edu
#############################


# import libraries
from SelfOrganizingMap import SelfOrganizingMap
import numpy as np
from pyts import datasets
import csv
from sklearn.metrics.cluster import adjusted_rand_score
import math
#import dtaidistance
#from pyts.metrics import dtw

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

# create output file to save results
fname = 'scores.csv'
headers = ['DatasetName' ,'n','length_time_series','Method','Total_DTW_Calls', 'dtwCallsPerEpoch', 'timePerEpoch', 'ARI' ,'Total_Time'
           ,'Speed-up' ,'PercentPruned']
fout = open(fname, "w")
csvw = csv.DictWriter(fout, fieldnames=headers)
csvw.writeheader()
fout.close()


# set epochs
epochs = 20

# Pick and compare performance on any dataset
UCR_DataSetNames = datasets.ucr_dataset_list()

# loop through datasets

#dset = datasets.fetch_ucr_dataset('HandOutlines')
#UCR_DataSetNames=['InsectEPGRegularTrain']
counter = 0
for datasetName in UCR_DataSetNames[counter:]:
    try:
        obj = datasets.fetch_ucr_dataset(dataset=datasetName)
    except:
        continue

    inputs = np.append(obj['data_train'], obj['data_test'], axis=0)
    labels = np.append(obj['target_train'], obj['target_test'])
    

    # skip varying length datasets
    if datasets.ucr_dataset_info(datasetName)['n_timestamps' ]== 'Varying' or datasets.ucr_dataset_info(datasetName)['n_timestamps' ] == 'Variable':
        continue
    # no noise handling
    if -1 in labels:
        continue

    total_time_steps = int(datasets.ucr_dataset_info(datasetName)['n_timestamps'])* (datasets.ucr_dataset_info(datasetName)['test_size']+datasets.ucr_dataset_info(datasetName)['train_size'])
    # set window size as number of time steps in window at 5% observation as the recommended best single window size (Paparizos & Gravano (2016,2017))
    windowSize = int(len(inputs[0] ) *0.05)
    K = len(np.unique(labels))

    # create mesh size -- small mesh size to replicate K-means for quicker execution when there is no need to visualize
    hiddenSize = [1, K]

    # list to save results
    rows = []
    print(str(counter) + '. ' + datasetName + ' in progress.')

    print('clustering..')
    # initilize SOM
    np.random.seed(0)
    SOM = SelfOrganizingMap(inputSize=len(inputs[0]), hiddenSize=hiddenSize)
    stats_dtw = SOM.iterate(inputs ,epochs = epochs ,windowSize = windowSize ,k=K ,randomInitilization=False, bounding = False)
    #SOM.plotMap(inputs,windowSize, labels=stats_dtw['labels'],path = 'plot_epoch')

    # reinitilize SOM so weights are reset
    np.random.seed(0)
    SOM = SelfOrganizingMap(inputSize=len(inputs[0]), hiddenSize=hiddenSize)
    stats_bounding = SOM.iterate(inputs, epochs=epochs, windowSize=windowSize, k=K,randomInitilization=False, bounding=True)

    # calculations

    dtw_Total_Time = np.sum(stats_dtw['timePerEpoch'])
    Total_DTW_Calls_unbounded = stats_dtw['Calculated']

    d = {}
    d['DatasetName'] = datasetName
    d['Method'] = 'Full DTW'
    d['Total_DTW_Calls'] = Total_DTW_Calls_unbounded
    d['dtwCallsPerEpoch'] = stats_dtw['dtwCallsPerEpoch']
    d['timePerEpoch'] = stats_dtw['timePerEpoch']
    d['ARI'] = adjusted_rand_score(labels_true = labels, labels_pred = stats_dtw['labels'] )
    d['Total_Time'] = dtw_Total_Time
    d['Speed-up'] = 'n/a'
    d['PercentPruned'] = 'n/a'
    d['n'] = len(inputs)
    d['length_time_series'] = datasets.ucr_dataset_info(datasetName)['n_timestamps' ]
    rows.append(d)
    print('DTW ARI is: '+str(d['ARI']))



    # calculations
    Total_DTW_Calls_bounded = stats_bounding['Calculated']
    percent_pruned = ((Total_DTW_Calls_unbounded - Total_DTW_Calls_bounded ) /Total_DTW_Calls_unbounded ) *100


    d = {}
    d['DatasetName'] = datasetName
    d['Method'] = 'Bounded DTW'
    d['Total_DTW_Calls'] = Total_DTW_Calls_bounded
    d['dtwCallsPerEpoch'] = stats_bounding['dtwCallsPerEpoch']
    d['timePerEpoch'] = stats_bounding['timePerEpoch']
    d['ARI'] = adjusted_rand_score(labels_true = labels, labels_pred = stats_bounding['labels'] )
    d['Total_Time'] = np.sum(stats_bounding['timePerEpoch'])
    d['Speed-up'] =  dtw_Total_Time /d['Total_Time']
    d['PercentPruned'] = percent_pruned
    d['n'] = len(inputs)
    d['length_time_series'] = datasets.ucr_dataset_info(datasetName)['n_timestamps']
    rows.append(d)
    print('DTW ARI is: ' + str(d['ARI']))
    # write both results to file
    fout = open(fname, "a")
    csvw = csv.DictWriter(fout, fieldnames=headers)
    csvw.writerows(rows)
    fout.close()


    print(str(counter) + '. ' + datasetName +' clustered.')
    counter += 1
