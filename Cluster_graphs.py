import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections



# Load HDF5 file
filename = 'E:\Marie\R65_dict_misor.dream3d'
f = h5py.File(filename, 'r')

# Loading global variables needed
#    Features
featEquivDiameter = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatureEquivalentDiameters'][:])
featureIDs = np.squeeze(f['DataContainers/ImageDataContainer/CellData/FeatureIds'][:])
featureARs = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/AspectRatios'][:])
ipfColors = np.squeeze(f['DataContainers/ImageDataContainer/CellData/IPFColor'][:])
featNeighborList = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatNeighborList'][:])
featNumNeighbors = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatNumNeighbors'][:])
featMisorList = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/FeatMisorientationList'][:])
featParentIDs = np.squeeze(f['DataContainers/ImageDataContainer/CellFeatureData/ParentIds'][:])
featNeighborIndex = np.cumsum(featNumNeighbors)
#   Clusters
trdEquivDiameter = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDEquivalentDiameters'][:])
trdNeighborList = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDNeighborList'][:])
trdNumNeighbors = np.squeeze(f['DataContainers/ImageDataContainer/TRDs/TRDNumNeighbors'][:])
trdNeighborIndex = np.cumsum(trdNumNeighbors)
#   Additional useful things
numberOfClusters = np.ma.size(trdEquivDiameter)
numberOfFeatures = np.ma.size(featEquivDiameter)

# output file to save results
output = open('graph_ppties_2.txt', 'w')

# Define on which clusters we want to do the analysis
cluster_list = [20220, 16300, 3201, 123, 13299, 3151, 23052, 2625, 20789, 22139, 7586, 19711, 11489, 22795, 8191, 4201,
                19043, 11064, 21082, 3155, 24061, 15061, 14627, 17036, 16265, 13591, 13321, 6139, 12764, 21833, 24879,
                15008, 5923, 24520, 17769, 17040, 5797, 4344, 5999, 19652, 6680, 1272, 11561, 4311, 6910, 4003, 6215,
                8762, 15866, 12982, 24064, 11675, 10758, 10373, 1254, 9324, 10035, 4568, 12734, 20710, 15765, 23405,
                9502, 18209, 23040, 18205, 11031, 7982, 14340, 2085, 609, 13638, 2564, 20707, 17146, 23120, 19206,
                18725, 11273, 17636, 2639, 5173, 2126, 12814, 2100, 18064, 23091, 23511, 6267, 12442, 3557, 21456, 6936,
                7818, 16588, 5282, 22789, 20125, 8936, 23370, 23021, 15311, 1191, 5179, 13039, 13788, 11813, 9941, 5539]
cluster_list_2 = [16171, 24180, 2117, 19614, 24888, 5321, 6838, 19229, 24074, 22619, 8523]

cluster  = 8992

def network_it(cluster):

    #  -- Features --
    # How many features belong to the cluster
    cluster_numfeat = 0
    for feature in range(0, numberOfFeatures):
        if featParentIDs[feature] == cluster:
            cluster_numfeat += 1
    # Get features IDs
    feature_ids = np.zeros(cluster_numfeat, dtype=int)
    init = 0
    for id in range(0, cluster_numfeat):
        for feat in range(init, numberOfFeatures):
            if featParentIDs[feat] == cluster:
                feature_ids[id] = feat
                break
        init = feature_ids[id]+1

    # -- Reconstructing twin clusters --
    # Get the neighbors of each feature
    edge_list = list()
    for i in range(0, int(cluster_numfeat)):
        feature = feature_ids[i]
        num_neighbors = featNeighborIndex[feature] - featNeighborIndex[feature-1]
        neighbor_list = np.copy(featNeighborList[featNeighborIndex[feature-1]:featNeighborIndex[feature]])
        neighbor_mis = np.copy(featMisorList[featNeighborIndex[feature-1]:featNeighborIndex[feature]])
        for neighbor in range(0, num_neighbors):
            if featParentIDs[neighbor_list[neighbor]] == cluster_id and 58.9 < neighbor_mis[neighbor] < 60:  # 60deg rotation around <111> directions +/- 2deg
                edge_list.append((feature, neighbor_list[neighbor]))
        # get data for the graph
        nodelist = feature_ids.tolist()
        network = nx.Graph()
        network.clear()
        network.clear()
        network.add_nodes_from(nodelist)
        network.add_edges_from(edge_list)
    return cluster_numfeat, nodelist, network


def pretty_plot(cluster):

    #  -- Features --
    # How many features belong to the cluster
    cluster_numfeat = 0
    for feature in range(0, numberOfFeatures):
        if featParentIDs[feature] == cluster:
            cluster_numfeat += 1

    # Get features IDs
    feature_ids = np.zeros(cluster_numfeat, dtype=int)
    init = 0
    for id in range(0, cluster_numfeat):
        for feat in range(init, numberOfFeatures):
            if featParentIDs[feat] == cluster:
                feature_ids[id] = feat
                break
        init = feature_ids[id]+1

    # Get features sizes
    feature_sizes = np.zeros(cluster_numfeat, dtype=float)
    for i in range(0, cluster_numfeat):
        feature_sizes[i] = featEquivDiameter[feature_ids[i]]
    nodesizes = feature_sizes*200

    # Get features ipf colors
    feature_ipfcolor = np.zeros((cluster_numfeat, 3), dtype=int)
    for i in range(0, cluster_numfeat):
        feature_ipfcolor[i] = fetch_ipf_color(feature_ids[i])
    feature_ipfcolor_norm = (feature_ipfcolor / 255)

    # -- Reconstructing twin clusters --
    # Get the neighbors of each feature
    edge_list = list()
    for i in range(0, int(cluster_numfeat)):
        feature = feature_ids[i]
        num_neighbors = featNeighborIndex[feature] - featNeighborIndex[feature-1]
        neighbor_list = np.copy(featNeighborList[featNeighborIndex[feature-1]:featNeighborIndex[feature]])
        neighbor_mis = np.copy(featMisorList[featNeighborIndex[feature-1]:featNeighborIndex[feature]])
        for neighbor in range(0, num_neighbors):
            if featParentIDs[neighbor_list[neighbor]] == cluster and 58.9 < neighbor_mis[neighbor] < 60:  # 60deg rotation around <111> directions +/- 2deg
                edge_list.append((feature, neighbor_list[neighbor]))

        # get data for the graph
        nodelist = feature_ids.tolist()
        nodesize = nodesizes.tolist()
        color_list = feature_ipfcolor_norm.tolist()
        graph = nx.Graph()
        graph.clear()
        graph.clear()
        graph.add_nodes_from(nodelist)
        graph.add_edges_from(edge_list)
        plt.figure()
        nx.draw_spring(graph, node_size=nodesize, with_labels=True, node_color=color_list,
                       font_size=12, font_color='k', alpha=0.7)
        plt.axis('off')
        plt.show()

    return cluster_numfeat, nodelist, graph


def fetch_ipf_color(feature_id):
    for i in range(0, 325):
        for j in range(0, 601):
            for k in range(0, 601):
                if featureIDs[i, j, k] == feature_id:
                    x = i
                    y = j
                    z = k
    color_list_rgb = ipfColors[x, y, z].tolist()
    print(color_list_rgb)
    return color_list_rgb


def density(g):
    d = nx.density(g)
    return d


def diameter(g):
    diam = nx.diameter(g)
    return diam


def clust_coeff(g, n):
    clust_coeffs = nx.clustering(g, n)
    return clust_coeffs


def btw_centr(g):
    btw = nx.eigenvector_centrality(network)
    nb_main_nodes = 0
    for value in btw.values():
        if value > 0.2:
            nb_main_nodes += 1
    return btw, nb_main_nodes


def average_clustering(g):
    av_clust = nx.average_clustering(g, count_zeros=True)
    return av_clust


def plot_degree_distribution(graph):

    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = sorted(nx.connected_component_subgraphs(graph), key=len, reverse=True)[0]
    pos = nx.spring_layout(graph)
    plt.axis('off')
    nx.draw_networkx_nodes(graph, pos, node_size=20)
    nx.draw_networkx_edges(graph, pos, alpha=0.4)


# Main program
output.write('clusterID \t number of features \t equivalent diameter \t density \t number of key nodes \t \n')
for i in range(0, np.ma.size(cluster_list_2)):
    cluster_id = cluster_list_2[i]
    number_features, node_list, network = network_it(cluster_id)
    eqdiam = trdEquivDiameter[cluster_id]
    # output.write('list of nodes' + str(node_list) + '\n')
    dens = density(network)
    # diam = diameter(network)
    # output.write('diameter: ' + str(diam) + '\n')
    # clustering_of_nodes = clust_coeff(network, len(node_list))
    # output.write('clustering coefficients of nodes:' + str(clustering_of_nodes) + '\n')
    count, number_nodes = btw_centr(network)
    output.write(repr(cluster_id) + '\t')
    output.write(str(number_features) + '\t')
    output.write(str(eqdiam) + '\t')
    output.write(str(dens) + '\t')
    output.write(str(count) + '\t')
    output.write(repr(number_nodes) + '\t \n')
    network.clear()
output.close()