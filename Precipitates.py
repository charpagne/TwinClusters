import h5py
import numpy as np
import matplotlib.pyplot as plt
import collections



# Load HDF5 file
filename = 'E:\Marie\R65_dict_lowerTol.dream3d'
f = h5py.File(filename, 'r')
# List all groups
# print("Keys: %s" % f.keys())

# Loading global variables needed
#    Features
featEquivDiameter = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatureEquivalentDiameters'][:])
featNeighborList = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatNeighborList'][:])
featNumNeighbors = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatNumNeighbors'][:])
featMisorList = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatMisorientationList'][:])
featParentIDs = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/ParentIds'][:])
# featEulers = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/AvEulers'][:])
featSharedSurfaceArea = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatSharedSurfaceAreaList'][:])
#   Twin Related Domains
trdEquivDiameter = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDEquivalentDiameters'][:])
trdNeighborList = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDNeighborList'][:])
trdNumNeighbors = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDNumNeighbors'][:])
trdVolumes = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDVolumes'][:])
# Calc global volume
# volume_tot = np.sum(trdVolumes)


#   Creating the NeighborIndexes lists
featNeighborIndex = np.cumsum(featNumNeighbors)
trdNeighborIndex = np.cumsum(trdNumNeighbors)

# Other useful quantities
numberOfFeatures = np.ma.size(featNumNeighbors)
numberOfClusters = np.ma.size(trdNumNeighbors)

# -- Twin Clusters --
# Find twin clusters
# Find matrix grains
critical_diam = 6
grain_TRDs = np.where(trdEquivDiameter > critical_diam)
grain_IDs = np.transpose(grain_TRDs)

# How many features per grain
TRDs_numTwinsuniq = np.unique(featParentIDs, return_counts=True)
TRDs_numTwins = np.squeeze(TRDs_numTwinsuniq)
grain_numTwins = np.zeros(np.ma.size(grain_TRDs), dtype=object)
for i in range(0, np.ma.size(grain_TRDs)):
    j = grain_TRDs[0][i]
    grain_numTwins[i] = TRDs_numTwins[1][j]

# -- Precipitates --
# Find precipitates
prec_TRDs = np.where(trdEquivDiameter <= critical_diam)

# -- Differentiation based on neighborhood --
# ---
# Zero neighbors
zero_neighbors = np.where(trdNumNeighbors == 0)
prec_zero_neighbors = np.intersect1d(zero_neighbors, prec_TRDs)

# ---
# One neighbor
trd_one_neighbor = np.where(trdNumNeighbors == 1)
prec_one_neighbor = np.intersect1d(trd_one_neighbor, prec_TRDs)
# ID of hosting cluster
host_prec_one_neighbor = np.zeros(np.ma.size(prec_one_neighbor), dtype=int)
for i in range(0, np.ma.size(prec_one_neighbor)):
    prec = prec_one_neighbor[i]  # prec = id of the current precipitate
    host_prec_one_neighbor[i] = trdNeighborIndex[prec-1]
# Get the features that correspond to those precipitates with only one neighbor
feat_one_neighbor = np.zeros(np.ma.size(trd_one_neighbor), dtype=int)
for i in range(0, np.ma.size(prec_one_neighbor)):
    j = prec_one_neighbor[i]
    for k in range(0, numberOfFeatures):
        if featParentIDs[k] == j:
            feat_one_neighbor[i] = k
            print i
            break
# find the strictly intragranular features
feat_intragranular = np.zeros_like(feat_one_neighbor)
for i in range(0, np.ma.size(feat_one_neighbor)):
    feature = feat_one_neighbor[i]
    if featNumNeighbors[feature] == 1:
        feat_intragranular[i] = feature
# delete zero values
feat_intragranular = feat_intragranular[feat_intragranular != 0]
# find the prec clusters that correspond to those intragranular features
prec_intragranular = np. zeros_like(feat_intragranular)
for i in range(0, np.ma.size(feat_intragranular)):
    feat = feat_intragranular[i]
    prec_intragranular[i] = featParentIDs[feat]
# get the precipitates that lay on the twin boundaries (by default)
prec_TBs = np.setdiff1d(prec_one_neighbor, prec_intragranular)



# ---
# Two or more neighbors
two_plus_neighbors = np.where(trdNumNeighbors >= 2)
prec_twoplus_neighbors = np.intersect1d(two_plus_neighbors, prec_TRDs)
# How many neighbors do they have
prec_twoplus_neighbors_numneigh = np.zeros(np.ma.size(prec_twoplus_neighbors), dtype=int)
for i in range(0, np.ma.size(prec_twoplus_neighbors)):
    j = prec_twoplus_neighbors[i]
    prec_twoplus_neighbors_numneigh[i] = trdNumNeighbors[j]
max_neigh = np.max(prec_twoplus_neighbors_numneigh)  # maximum number of neighbors
prec_twoplus_neighbors_neighIDs = np.zeros((np.ma.size(prec_twoplus_neighbors), 27), dtype=int)
for i in range(0, np.ma.size(prec_twoplus_neighbors)):
    precipitate = prec_twoplus_neighbors[i]
    rank_start = trdNeighborIndex[precipitate-1]
    rank_finish = trdNeighborIndex[precipitate]-1
    num = rank_finish - rank_start
    prec_twoplus_neighbors_neighIDs[i][0:num] = trdNeighborList[rank_start:rank_finish]
# find the precipitates that lay on the twin boundaries of those neighbors
feat_twoplus_neighbors = np.zeros(np.ma.size(prec_twoplus_neighbors), dtype=int)
# retrieve the features that correspond to those precipitate trds
for i in range(0, np.ma.size(prec_twoplus_neighbors)):
    j = prec_twoplus_neighbors[i]
    for k in range(0, np.ma.size(featEquivDiameter)):
        if featParentIDs[k] == j:
            feat_twoplus_neighbors[i] = k
            print i
            break
feat_twoplus_neighbors_numneigh = np.zeros(np.ma.size(feat_twoplus_neighbors), dtype=int)
# get nber of neighbors of those features
for i in range(0, np.ma.size(prec_twoplus_neighbors)):
    j = feat_twoplus_neighbors[i]
    feat_twoplus_neighbors_numneigh[i] = featNumNeighbors[j]

# get the neighboring features to those precipitate features
feat_twoplus_neighbors_featneighIDs = np.zeros((np.ma.size(feat_twoplus_neighbors), 27), dtype=int)
for i in range(0, np.ma.size(feat_twoplus_neighbors)):
    feat = feat_twoplus_neighbors[i]
    rank_start = featNeighborIndex[feat-1]
    rank_finish = featNeighborIndex[feat]-1
    num = rank_finish - rank_start
    feat_twoplus_neighbors_featneighIDs[i][0:num] = featNeighborList[rank_start:rank_finish]

# get the parent to which those neighbor features belong
feat_twoplus_neighbors_parneighIDs = np.zeros_like(feat_twoplus_neighbors_featneighIDs, dtype=int)
for i in range(0, (np.ma.size(feat_twoplus_neighbors))):
    for j in range(0, 27):
        var = feat_twoplus_neighbors_featneighIDs[i][j]
        feat_twoplus_neighbors_parneighIDs[i][j] = featParentIDs[var]

# check if the parent ID appears twice or more
feat_onTBGBs = np.zeros((np.ma.size(feat_twoplus_neighbors), 1), dtype=int)
parent_ofTBGBsNeighbor = np.zeros(np.ma.size(feat_twoplus_neighbors), dtype=int)
for i in range(0, np.ma.size(feat_twoplus_neighbors)):
    print i
    list_of_parents = feat_twoplus_neighbors_parneighIDs[i]
    for j in range(0, 27):
        parent = list_of_parents[j]
        occurrence = np.count_nonzero(list_of_parents == parent)
        if occurrence >= 2:
            feat_onTBGBs[i] = feat_twoplus_neighbors[i]
            parent_ofTBGBsNeighbor[i] = parent
            break
            i += 1
        elif occurrence <= 1:
            i += 1

# remove zero values
feat_onTBGBs = feat_onTBGBs[feat_onTBGBs != 0]
parent_ofTBGBsNeighbor = parent_ofTBGBsNeighbor[parent_ofTBGBsNeighbor != 0]
# retrieve IDs of the clusters that correspond to the TBGB features
prec_TBGBs = np.zeros_like(feat_onTBGBs, dtype=int)
for i in range(0, np.ma.size(feat_onTBGBs)):
    feat = feat_onTBGBs[i]
    prec_TBGBs[i] = featParentIDs[feat]


# get the precipitates that lay on GBs by difference of previous arrays
prec_GBs = np.setdiff1d(prec_twoplus_neighbors, prec_TBGBs)
# remove the zero values in case there are any
prec_GBs = prec_GBs[prec_GBs != 0]


# -- get number of precipitates of each category
nberGBprec = np.ma.size(prec_GBs)
nberTBGBprec = np.ma.size(prec_TBGBs)
nberIntraprec = np.ma.size(prec_intragranular)
nberTBprec = np.ma.size(prec_TBs)





# Get all the data for each cluster
perClusterCounts = open('perClusterCounts.txt', 'w')
perClusterData = open('perClusterData.txt', 'w')

perClusterCounts.write('cluster ID \t size \t intragranular \t num_features \t prec \t onGBs \t onTBGBs \t onTBs \n \n')
perClusterData.write('clusterID \t size \t num_features \t intragranular prec \t onGBs \t on TBGBs \t onTBs \n \n')
for i in range(0, np.ma.size(grain_TRDs)):
    trd = grain_TRDs[0][i]
    size = trdEquivDiameter[trd]
    numfeatures = TRDs_numTwins[1][trd]
    volume = trdVolumes[trd]
    startIndex = trdNeighborIndex[trd-1]
    finishIndex = trdNeighborIndex[trd]-1
    arrayOfNeighbors = trdNeighborList[startIndex:finishIndex]
    arrayOfPrecNeigh = np.intersect1d(prec_TRDs, arrayOfNeighbors)  # contains all neighboring precipitates to each TRD
    # get all precipitates IDs
    arrayOfPrecIntra = np.intersect1d(prec_intragranular, arrayOfNeighbors)
    arrayOfPrecGBs = np.intersect1d(prec_GBs, arrayOfNeighbors)
    arrayOfPrecTBGBs = np.intersect1d(prec_TBGBs, arrayOfNeighbors)
    arrayOfPrecTBs = np.intersect1d(prec_TBs, arrayOfNeighbors)
    perClusterCounts.write(repr(trd) + '\t' + repr(size) + '\t' + repr(numfeatures) + '\t' + repr(np.ma.size(arrayOfPrecIntra)) + '\t' + repr(np.ma.size(arrayOfPrecGBs)) + '\t' + repr(np.ma.size(arrayOfPrecTBGBs)) + '\t' + repr(np.ma.size(arrayOfPrecTBs)) + '\n')
    perClusterData.write(repr(trd) + '\t' + repr(size) + '\t' + repr(numfeatures) + '\t' + repr(arrayOfPrecIntra) + '\n' + repr(arrayOfPrecGBs) + '\n' + repr(arrayOfPrecTBGBs) + '\n' + repr(arrayOfPrecTBs) + '\n \n')
perClusterData.close()
perClusterCounts.close()



# Get data for a given cluster
cluster_id = 19711
index = 1012

np.savetxt('grainTRDs.txt', grain_TRDs, fmt='%d')
# get its neighbors
startIndex = trdNeighborIndex[cluster_id-1]
finishIndex = trdNeighborIndex[cluster_id]-1
arrayOfNeighbors = trdNeighborList[startIndex:finishIndex]
arrayOfPrecNeigh = np.intersect1d(prec_TRDs, arrayOfNeighbors)

featuresOfCluster = [1472, 36453, 21122, 18316, 14389, 11467, 13829, 38501, 27209, 28374, 41603, 4692, 4921, 12725, 40688]

# intragranular precipitates
arrayOfPrecIntra = np.intersect1d(prec_intragranular, arrayOfPrecNeigh)  # indexes of intragranular precipitates
# get the corresponding features
arrayOfFeatIntra = np.zeros_like(arrayOfPrecIntra)
for i in range(0, np.ma.size(arrayOfFeatIntra)):
    j = arrayOfPrecIntra[i]
    for k in range(0, numberOfFeatures):
        if featParentIDs[k] == j:
            arrayOfFeatIntra[i] = k
            print i
            break
arrayOfHostingFeatures = np.zeros_like(arrayOfFeatIntra)  # hosting features to intragr precipitates
for i in range(0, np.ma.size(arrayOfPrecIntra)):
    prec = arrayOfFeatIntra[i]
    index = featNeighborIndex[prec-1]
    arrayOfHostingFeatures[i] = featNeighborList[index]

for i in range(0, np.ma.size(arrayOfPrecIntra)):
    feature = arrayOfFeatIntra[i]
    index = featNeighborIndex[feature-1]
    arrayOfHostingFeatures[i] = featNeighborList[index]


# precipitates on GBs
arrayOfPrecGBs = np.intersect1d(prec_GBs, arrayOfNeighbors)  # indexes of precipitates on grain boundaries
# get corresponding feature
arrayOfFeatGBs = np.zeros(np.ma.size(arrayOfPrecGBs), dtype=int)
for i in range(0, np.ma.size(arrayOfPrecGBs)):
    j = arrayOfPrecGBs[i]
    for k in range(0, numberOfFeatures):
        if featParentIDs[k] == j:
            arrayOfFeatGBs[i] = k
            print i
            break
# get the neighbor features to a given feature ID
feature_id = feat
rank_start = featNeighborIndex[feature_id-1]
rank_finish = featNeighborIndex[feature_id]-1
num = rank_finish - rank_start
feature_neighbors = featNeighborList[rank_start:rank_finish]
# get the features that belong to the cluster of interest
neighboringClusterFeatures = np.intersect1d(featuresOfCluster, feature_neighbors)

# precipitates on both TBs and GBs
arrayOfPrecTBGBs = np.intersect1d(prec_TBGBs, arrayOfNeighbors)  # indexes of prec in TB and GBs
# retrieve the corresponding features
arrayOfFeatTBGBs = np.zeros_like(arrayOfPrecTBGBs)
for i in range(0, np.ma.size(arrayOfPrecTBGBs)):
    j = arrayOfPrecTBGBs[i]
    for k in range(0, numberOfFeatures):
        if featParentIDs[k] == j:
            arrayOfFeatTBGBs[i] = k
            print i
            break
# how many neighbors
precTBGBnumNeighbors = np.zeros_like(arrayOfPrecTBGBs)
sum = 0
for i in range(0, np.ma.size(arrayOfPrecTBGBs)):
    prec = arrayOfFeatTBGBs[i]
    precTBGBnumNeighbors[i] = featNumNeighbors[prec]
    sum += featNumNeighbors[prec]
precTBGBneigh = np.zeros((np.ma.size(arrayOfPrecTBGBs), np.max(precTBGBnumNeighbors)), dtype = int)
for i in range(0, np.ma.size(arrayOfPrecTBGBs)):
    prec = arrayOfFeatTBGBs[i]
    numNeighbors = precTBGBnumNeighbors[prec]
    for j in range(0, numNeighbors):
        precTBGBneigh[i][j] = featNeighborList[prec-1:prec-1+numNeighbors]


# find their neighbors
feature_id = feat
rank_start = featNeighborIndex[feature_id-1]
rank_finish = featNeighborIndex[feature_id]-1
num = rank_finish - rank_start
feature_neighborsTBGB = featNeighborList[rank_start:rank_finish]
# get the neighbor features that belong to the cluster
neighboringClusterFeaturesTBGBs = np.intersect1d(featuresOfCluster, feature_neighborsTBGB)


arrayOfPrecTBs = np.intersect1d(prec_TBs, arrayOfNeighbors)  # indexes of precipitates on TBs
arrayOfFeatTBs = np.zeros_like(arrayOfPrecTBs)
for i in range(0, np.ma.size(arrayOfPrecTBs)):
    j = arrayOfPrecTBs[i]
    for k in range(0, numberOfFeatures):
        if featParentIDs[k] == j:
            arrayOfFeatTBs[i] = k
            print i
            break

