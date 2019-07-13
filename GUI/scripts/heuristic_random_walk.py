import time
import sys
sys.path.append('scripts/tsplib-parser')
import TsplibParser
import numpy as np
import json
import random
from scipy.spatial import distance



estimated_cycle = [0]



def heuristic_random_walk(adj_mat, i):
    time_begin = time.process_time_ns()  # Starting timer
    heuristic_execution = heuristic_random_walk_recursive(adj_mat, i)
    time_end = time.process_time_ns()  # Stopping timer

    path = heuristic_execution[0]
    cost = heuristic_execution[1]
    time_exec = time_end - time_begin

    return path, cost, time_exec


def heuristic_random_walk_recursive(adj_mat, i):
    global estimated_cycle
    estimated_cycle.append(i)
    last_city = len(adj_mat) - 1

    if len(estimated_cycle) == len(adj_mat) + 1:
        estimated_cycle[0] = estimated_cycle[0] + adj_mat[estimated_cycle[1]][i]
        return estimated_cycle[1:], estimated_cycle[0]

    next_random_city = random_other_than(estimated_cycle[1:], last_city)
    estimated_cycle[0] += adj_mat[i][next_random_city]

    return heuristic_random_walk_recursive(adj_mat, next_random_city)


def random_other_than(alist, top):
    n = random.randint(0, top)
    while alist.__contains__(n):
        n = random.randint(0, top)

    return n


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

# coords = instance.get_nodes_coord()
# adj_mat = coord_to_adj_matrix(coords)

adj_mat = instance.get_adj_matrix().tolist()

random_start_node = random.randrange(len(adj_mat))

heuristic_execution = heuristic_random_walk(adj_mat, random_start_node)
heuristic_path = heuristic_execution[0]
heuristic_cost = heuristic_execution[1]
heuristic_time = heuristic_execution[2]

print(json.dumps({'execTime': heuristic_time, 'pathCost': heuristic_cost}, separators=(',', ': ')))
sys.stdout.flush()

