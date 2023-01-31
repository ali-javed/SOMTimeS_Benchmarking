#python libraries needed in code
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
from joblib import Parallel, delayed
from dtaidistance import dtw
from tslearn import metrics
from scipy.spatial.distance import cdist
import copy
from sklearn.cluster import SpectralClustering


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a,dtype=np.float64)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)

class SelfOrganizingMap:
    def __init__(self, inputSize, hiddenSize):
        ##############################
        # author: Ali Javed
        # October 14 2020
        # email: ajaved@uvm.edu
        #############################
        # Description: Class initilizer. This function creates an instance of neural network saving the parameters and setting random weights
        # inputsSize: number of input nodes needed i.e. 5.
        # hiddenSize: number of hidden layer nodes [2,3] will create a 2x3 node grid
        ########################################
        #set random see for reproducibility
        #np.random.seed(0)
        # initilize variables
        self.hiddenSize = np.asarray(hiddenSize)
        self.inputSize = inputSize
        # always start learning rate at 0.9
        self.learningRateInitial = 0.9
        self.learningRate = 0.9
        self.neighborhoodSizeInitial = int(self.hiddenSize[0] / 2)
        self.neighborhoodSize = int(self.hiddenSize[0] / 2)
        self.Umatrix = np.zeros((self.hiddenSize[0],self.hiddenSize[1]))

        
        # initilize weights between 0 and 1 for a 3d weights matrix
        self.weights_Kohonen = np.random.rand(self.hiddenSize[0]*self.hiddenSize[1], self.inputSize)

    def update_parameters(self, iteration, epoch):
        if self.neighborhoodSize!=0:
            self.neighborhoodSize = np.ceil(self.neighborhoodSizeInitial * (1 - (iteration / epoch)))
        self.learningRate = self.learningRateInitial * (1 - (iteration / epoch))

    def find_neighbor_indices(self, i, j):
        # function finds the neighboring rows and columns to include
        # i : i-th index
        # j : j-th index
        # dist: how big the neighborhood should span
        #########################################################
        rows = []
        columns = []

        # python indexing starts with 0 so adjust here
        i = i + 1
        j = j + 1
        if i > self.hiddenSize[0] or i < 1 or j > self.hiddenSize[1] or j < 1:
            neighborhood = set()
            return neighborhood
            
            
        rows = np.arange(i - int(self.neighborhoodSize), i + int(self.neighborhoodSize) + 1)
        columns = np.arange(j - int(self.neighborhoodSize), j + int(self.neighborhoodSize) + 1)

        # get neighbor indexes as a combination of rows and columns
        neighborhood = set()
        for row in rows:
            for column in columns:
                
                row = row % self.hiddenSize[0]

                column = column % self.hiddenSize[1]

                if row == 0:
                    row = self.hiddenSize[0]
                if column == 0:
                    column = self.hiddenSize[1]

                # do not update actual row, because it is used in the loop
                row_temp = row - 1
                column_temp = column - 1

                neighborhood.add((row_temp, column_temp))

        return neighborhood

    def bmu(self, x,windowSize,candidateWeights):
        ## function to find winning node
        #: input observatiopn

        # format input for use in this function --- dtw distance
        # x = np.reshape(x[0], (1, 1, len(x[0])))

        ####################################
        # calculate distances (in Euclidean and DTW it is the minimum). Iterate over all nodes to find distance
        #x = np.reshape(x,(1,len(x[0])))
        min_distance = float('inf')
        dtw_Cals = 0
        for i in candidateWeights:

            #get candidate weight
            #this needs to be a deep copy for dtw distance not to throw an error.
            weights = copy.deepcopy(self.weights_Kohonen[i])
            xCopy = copy.deepcopy(x)
            #get dtw distance
            distance = dtw.distance_fast(xCopy,weights, window = windowSize, use_pruning=False)
            
            dtw_Cals+=1
            #update min distance if new distance is lower
            if distance<=min_distance:
                bmuIndex = i
                min_distance = distance


        return [bmuIndex,dtw_Cals]





    def propogateForward(self,inputs,windowSize):
        ############
        # Description: Function forward propogates from input to grid
        # x: single input

        ##############################
        # input to Kohonen
        ##############################

        # make sure x is in correct shape for matrix multiplication
        #x = np.reshape(x, (1, len(x)))

        # find minimum upper bound as possible candidate for BMU
        #sqEucdistances = cdist(inputs, self.weights_Kohonen, 'sqeuclidean')
        Eucdistances = cdist(inputs, self.weights_Kohonen, 'euclidean')
        upper_bounds = Eucdistances.min(axis=1)
        bmus = []
        dtw_Cals = 0


        for i in range(0,len(inputs)):
            # calculate lower bounds
            lower_bounds = []
            for j in range(0, len(self.weights_Kohonen)):
                lower_bounds.append(metrics.lb_keogh(inputs[i], self.weights_Kohonen[j], radius=windowSize))

            lower_bounds = np.asarray(lower_bounds)
            #bmu is the min node based on sq euc distance

            #if min Euclidean distance between observation and all neurons is less than or equal to the minimum lower bound then node with minimum euclidean distance is the best matching neuron., 
            if upper_bounds[i]<=min(lower_bounds):
                bmus.append(np.argmin(Eucdistances[i]))


            #lower bounds are greater than bmu based on squared euclidean distance
            else:
                #still no need to calculate dtw with all nodes
                candidateWeights = np.argwhere(lower_bounds < upper_bounds[i]).flatten()

                #get best matching unit
                [bmu,calls] = self.bmu(inputs[i], windowSize, candidateWeights)
                bmus.append(bmu)
                dtw_Cals +=calls


        return bmus,dtw_Cals



    def update_weights_Kohonen(self, x,index):
        ############
        # Description: Function updates the Kohonen layer (SOM layer) after one forward pass (i.e., forwardPropogate)
        # x: single input
        #############

        if self.neighborhoodSize!=0:
            [r,c] = np.unravel_index(index, self.hiddenSize)
            neighborhood2d = self.find_neighbor_indices(r, c)
        
            neighborhood = []
            for neighbor in neighborhood2d:
                neighborhood.append(np.ravel_multi_index(neighbor,self.hiddenSize))
        else:
            neighborhood = [index]

        # implement update formula to update all neighborhood
        for neighbor in neighborhood:
            # calculate the update

            update = self.learningRate * (
                x.flatten() - self.weights_Kohonen[neighbor].flatten())

            # update the weights

            self.weights_Kohonen[neighbor] = self.weights_Kohonen[neighbor] + update


    def iterate(self, inputs, epochs, windowSize = 0,k = 1,randomInitilization=True):
        ############
        # Description: Function iterates to organize the Kohonen layer

        # inputs: all inputs
        # epochs: epochs to iterate for
        # path: Path to save SOM plots
        # windowSize: windowSize to be used by DTW (for project), not usefull in assignment and set to 0.
        # path: path to save SOM plot
        # labels: if ground truth is available for color coding plot
        # observationIDs: if observationIDs are available for assigning to outout

        ####
        #if hardClusters required
        observationNum = np.arange(0,len(inputs))

        #####
        
        # initilize weights between 0 and 1 for a 3d weights matrix
        #self.weights_Kohonen = np.zeros((self.hiddenSize[0]*self.hiddenSize[1], self.inputSize))


        #comment out to not use formula for weights initilization

        #######################
        # formula for weights initilization
        if randomInitilization:
            for i in range(0, len(inputs[0])):
                min_test =np.min(np.asarray(inputs)[:, i])
                max_test =np.max(np.asarray(inputs)[:, i])
                std_test = np.std(np.asarray(inputs)[:, i])
                #std_test = std_test/5
                firstPart = (np.mean(np.asarray(inputs)[:, i]) + np.random.uniform(std_test*-1, std_test))
                secondPart = (np.mean(np.asarray(inputs)[:, i])) * np.random.uniform(low=min_test, high=max_test, size=(len(self.weights_Kohonen)))
                weights = firstPart * secondPart
                self.weights_Kohonen[:, i] = weights



        num_cores = multiprocessing.cpu_count()

        inputs = np.asarray(inputs)

        dtwCalls_per_Epoch = []
        time_per_Epoch = []
        for epoch in range(0, epochs):

            # for each input
            #randomly shuffle for each epoch
            epoch_start = time.time()
            p = np.random.permutation(len(inputs))
            inputs = inputs[p]
            observationNum = observationNum[p]

            #parallell process block
            #######################
            #limitation of library require smaller jobs if dataset is too large
            inputs_p = np.array_split(inputs, num_cores)
            returnValues = Parallel(n_jobs=num_cores)(delayed(self.propogateForward)(i,windowSize) for i in inputs_p)
            returnValues = np.asarray(returnValues)
            #parse return values
            bmuIndices = np.asarray(returnValues[:,0])
            calls = returnValues[:,1]
            calls = sum(calls)
            bmuIndices = [item for sublist in bmuIndices for item in sublist]
            ######################

            #serial processing line
            #bmuIndices,calls = self.propogateForward(inputs,windowSize)

                

            #update all weights
            for j in range(0,len(bmuIndices)):
                self.update_weights_Kohonen(inputs[j], bmuIndices[j])

            dtwCalls_per_Epoch.append(calls)
            time_per_Epoch.append(time.time() - epoch_start)
            if epoch %20 == 0:
                #self.plotMap_RGB(inputs, windowSize=0, labels=inputs, path='../tutorial/RGB_' + str(epoch))
                print('Epoch : ' + str(epoch) + ' complete.')
                #print("--- %s seconds ---" % (time.time() - start_time))
                print('**************************************')


            self.update_parameters(epoch, epochs)
            if self.neighborhoodSize < 0.3:
                self.neighborhoodSize = 0





        sortingOrder = np.argsort(observationNum)
        bmuIndices = np.asarray(bmuIndices)
        bmuIndices = bmuIndices[sortingOrder]
        stats = {}
        stats['Calculated'] = sum(dtwCalls_per_Epoch)
        stats['dtwCallsPerEpoch'] = dtwCalls_per_Epoch
        stats['timePerEpoch'] = time_per_Epoch


        if k>1:

            distance_matrix = np.zeros((len(self.weights_Kohonen),len(self.weights_Kohonen)))
            for a in range(0,len(self.weights_Kohonen)):
                for b in range(a+1,len(self.weights_Kohonen)):
                    distance_matrix[a,b] = dtw.distance_fast(self.weights_Kohonen[a], self.weights_Kohonen[b], window=windowSize, use_pruning=False)
                    distance_matrix[b,a] = distance_matrix[a,b]
            distance_matrix = np.asarray(distance_matrix)


            min_val = np.min(distance_matrix.flatten())
            max_val = np.max(distance_matrix.flatten())

            normaize_distance_matrix = (distance_matrix - min_val) / (max_val - min_val)
            similarity_matrix = 1 - normaize_distance_matrix
            clustering = SpectralClustering(n_clusters=k, n_init=100, assign_labels='discretize',random_state=0,affinity = 'precomputed').fit(similarity_matrix)

            labels = clustering.labels_


            node_to_cluster = {}

            for i in range(0,len(labels)):
                node_to_cluster[i] = labels[i]

            predictions = []
            for i in range(0,len(bmuIndices)):
                predictions.append(node_to_cluster[bmuIndices[i]])

            stats['labels'] = predictions
        else:
            stats['labels'] = []

        #we could simply do this but for a larger grid size this will not work nor recommended
        #stats['labels'] = bmuIndices

        return stats



        
    def createUmatrix(self,windowSize):


        normalizedWeights = self.weights_Kohonen
        self.neighborhoodSize = 1
        self.Umatrix = np.zeros((self.hiddenSize[0], self.hiddenSize[1]))

        for idx in range(0,len(self.weights_Kohonen)):
            nodeWeights = normalizedWeights[idx]

            [i, j] = np.unravel_index(idx, self.hiddenSize)
            #find all the neighbors for node at i,j
            neighbors2d = self.find_neighbor_indices(i, j)
            #remove self
            neighbors2d.remove((i, j))
            #get weights for node at i,j

            neighbors = []
            for neighbor in neighbors2d:
                neighbors.append(np.ravel_multi_index(neighbor, self.hiddenSize))

            for neighbor in neighbors:

                #for dtw
                neighborWeights = normalizedWeights[neighbor]
                distance = dtw.distance_fast(neighborWeights, nodeWeights, window=windowSize, use_pruning=False)

                self.Umatrix[i,j] += distance

        return self.Umatrix

                    
                
    def locationOnMap(self,inputs,windowSize):
        locations = []
        inputs = np.asarray(inputs)
        bmuIndices, dtwCalls = self.propogateForward(inputs, windowSize)


        for idx in bmuIndices:
            [r, c] = np.unravel_index(idx, self.hiddenSize)
            d = {}
            d['x'] = r
            d['y'] = c
            locations.append(d)
        return locations




    def plotMap(self, inputs,windowSize, labels=[], path = 'plot_epoch'):


        #colors to label points
        inputs=np.asarray(inputs)
        colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']

        # plot observations with labels
        plt.figure(figsize = (6,6))
        

        plt.imshow(self.Umatrix.transpose(), cmap='Greys',alpha=1)

        NodeSize = 20
        #assign labels
        if len(labels)== 0:
            labels = np.zeros(len(inputs))

        #get unique classes
        uniqueLabels = set(labels)
        if len(uniqueLabels)>8:
            print('Warning: Maximum number of classes for color coding is 5. No coloring will be used.')
            labels = np.zeros(len(inputs))
            uniqueLabels = set(labels)

        for label in uniqueLabels:
            label = int(label)
            #find index based on ground truth
            idx = np.argwhere(np.asarray(labels) == label).flatten()
            inputs_p = inputs[idx.flatten()]

            bmuIndices,dtwCalls = self.propogateForward(inputs_p,windowSize)

            x = []
            y = []
            for idx in bmuIndices:
                [r,c] = np.unravel_index(idx, self.hiddenSize)

                rand_num = np.random.uniform(0, 0.45, 2)
                x.append(r+rand_num[0])
                y.append(c+rand_num[1])

            plt.scatter(x, y, s=NodeSize, color=colors[label],label = label,alpha=1)



        
        plt.xlim(0 - 5, self.hiddenSize[0] + 5)
        plt.ylim(0 - 5, self.hiddenSize[1] + 5)
        plt.xlabel('Nodes', fontsize=22)
        plt.ylabel('Nodes', fontsize=22)
        #plt.xticks([])
        #plt.yticks([])
        if len(uniqueLabels)>1:
            plt.legend(fontsize = 18,bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.savefig(path+'.png',bbox_inches = 'tight')

        return True

    
    def getWeights(self):
        return self.weights_Kohonen


    def getUmatrix(self):
        return self.Umatrix

    def saveUmatrix(self,path):
        np.save(path, self.Umatrix)


    def saveWeights(self,path):
        np.save(path,self.weights_Kohonen)

    def loadWeights(self,path):
        self.weights_Kohonen = np.load(path)
