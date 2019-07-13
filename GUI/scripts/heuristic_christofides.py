import time
import sys
from pytsp import christofides_tsp
sys.path.append('scripts/tsplib-parser')
import TsplibParser
import numpy as np
import json
from scipy.spatial import distance




def heuristic_christofides(mat):
    time_begin = time.process_time_ns()  # Starting timer
    path = christofides_tsp.christofides_tsp(mat)
    time_end = time.process_time_ns()  # Stopping timer

    cost = calculate_cost_from_path(mat, path)
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


params=json.loads(sys.argv[1])
instance = TsplibParser.load_instance("scripts/tsp-dataset/"+params["fileName"],None)

coords = instance.get_nodes_coord()
adj_mat = coord_to_adj_matrix(coords)

# adj_mat = instance.get_adj_matrix()


heuristic_execution = heuristic_christofides(adj_mat)
heuristic_path = heuristic_execution[0]
heuristic_cost = heuristic_execution[1]
heuristic_time = heuristic_execution[2]

print(json.dumps({'execTime': heuristic_time, 'pathCost': heuristic_cost}, separators=(',', ': ')))
sys.stdout.flush()

