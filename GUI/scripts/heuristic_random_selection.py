import time
import random
import sys
import numpy as np
import json

# from pandas.tests.extension.numpy_.test_numpy_nested import np
# from scipy.spatial import distance
from scipy.spatial import distance


def heuristic_random_selection(g):
    path = []

    time_begin = time.time()  # Starting timer
    for i in range(len(g)):
        path.append(i)
    random.shuffle(path)
    cost = calculate_cost_from_path(g, path)
    time_end = time.time()  # Stopping timer

    time_exec = time_end - time_begin

    return path, cost, time_exec


def calculate_cost_from_path(adj_mat, path):
    # Path is noncyclic

    nb_nodes = len(adj_mat)
    cost = adj_mat[path[0]][path[nb_nodes - 1]]
    for i in range(nb_nodes - 1):
        cost += adj_mat[path[i]][path[i+1]]

    return cost


def coord_to_adj_matrix(coord):
    num_nodes = len(coord)
    dist_mat = np.zeros((num_nodes, num_nodes))
    for _ in range(0, num_nodes):
        for i in range(_ + 1, num_nodes):
            a = (coord[_][0], coord[_][1])
            b = (coord[i][0], coord[i][1])
            dist_mat[_][i] = int(distance.euclidean(a, b))

    dist_mat += dist_mat.T

    return dist_mat


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

sys.path.append('scripts/tsplib-parser')
import TsplibParser
params=json.loads(sys.argv[1])

instance = TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)

coords = instance.get_nodes_coord()
adj_mat = coord_to_adj_matrix(coords)

heuristic_execution = heuristic_random_selection(adj_mat)
heuristic_path = heuristic_execution[0]
heuristic_cost = heuristic_execution[1]
heuristic_time = heuristic_execution[2]

print(json.dumps({'execTime': heuristic_time, 'pathCost': heuristic_cost}, separators=(',', ': ')))
sys.stdout.flush()

