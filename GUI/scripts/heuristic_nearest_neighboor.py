import time
import sys
sys.path.append('scripts/tsplib-parser')
import TsplibParser
import numpy as np
import json
from scipy.spatial import distance



estimated_cycle = [0]


def heuristic_nearest_neighboor(adj_mat, i):
    time_begin = time.process_time_ns()  # Starting timer
    heuristic_exec = heuristic_nearest_neighboor_recursive(g, i)
    path = heuristic_exec[0]
    cost = heuristic_exec[1]
    time_end = time.process_time_ns()  # Stopping timer

    time_exec = time_end - time_begin

    return path, cost, time_exec


def heuristic_nearest_neighboor_recursive(adj_mat, i):
    global estimated_cycle
    estimated_cycle.append(i)
    if len(estimated_cycle) == len(adj_mat) + 1:
        estimated_cycle[0] += adj_mat[estimated_cycle[1]][i]
        return estimated_cycle[1:], estimated_cycle[0]
    estimated_cycle[0] = estimated_cycle[0] + min_adj_cost(adj_mat, i)[0]
    return heuristic_nearest_neighboor_recursive(adj_mat, min_adj_cost(adj_mat, i)[1])


def min_adj_cost(mat, node):
    adj_weights = []

    min_cost = -1
    min_cost_node = -1

    for i in range(len(mat)):
        if i == node:
            continue
        estimated_cycle[0] = estimated_cycle[0] - 500
        if estimated_cycle.count(i) == 0:
            weight = mat[node][i]
            adj_weights.append(weight)
            if min_cost == -1 or weight < min_cost:
                min_cost = weight
                min_cost_node = i

        estimated_cycle[0] = estimated_cycle[0] + 500

    min_cost = min(adj_weights)

    return min_cost, min_cost_node


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


params=json.loads(sys.argv[1])
instance = TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)

coords = instance.get_nodes_coord()
adj_mat = coord_to_adj_matrix(coords)

heuristic_execution = heuristic_nearest_neighboor(adj_mat)
heuristic_path = heuristic_execution[0]
heuristic_cost = heuristic_execution[1]
heuristic_time = heuristic_execution[2]

print(json.dumps({'execTime': heuristic_time, 'pathCost': heuristic_cost}, separators=(',', ': ')))
sys.stdout.flush()

